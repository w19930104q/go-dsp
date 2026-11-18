/*
 * Copyright (c) 2012 Madelynn Blue <blue.mlynn@gmail.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

// Package wav provides support for the WAV file format.
//
// Supported formats are PCM (8, 16, 24 and 32 bits per sample) and IEEE
// float (32 and 64 bits per sample), with any number of interleaved
// channels. Extended chunks (JUNK, bext, and others added by tools like
// ProTools) are ignored.
//
// Samples are always returned (and accepted) as one interleaved slice,
// channel-major within each frame: frame i of channel c lives at
// i*NumChannels+c. Deinterleave/Interleave split and merge channels.
package wav

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"time"
)

const (
	wavFormatPCM         = 1
	wavFormatIEEEFloat   = 3
	wavFormatExtensible  = 0xfffe
	fmtExtensibleMinSize = 40
)

// Header contains Wav fmt chunk data.
type Header struct {
	AudioFormat   uint16
	NumChannels   uint16
	SampleRate    uint32
	ByteRate      uint32
	BlockAlign    uint16
	BitsPerSample uint16
}

// Wav reads wav files.
type Wav struct {
	Header
	// Samples is the total number of available samples.
	Samples int
	// Duration is the estimated duration based on reported samples.
	Duration time.Duration

	r io.Reader
}

// New reads the WAV header from r.
func New(r io.Reader) (*Wav, error) {
	var w Wav
	header := make([]byte, 16)
	if _, err := io.ReadFull(r, header[:12]); err != nil {
		return nil, err
	}
	if string(header[0:4]) != "RIFF" {
		return nil, fmt.Errorf("wav: missing RIFF")
	}
	if string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("wav: missing WAVE")
	}
	hasFmt := false
	for {
		if _, err := io.ReadFull(r, header[:8]); err != nil {
			return nil, err
		}
		sz := binary.LittleEndian.Uint32(header[4:])
		switch typ := string(header[:4]); typ {
		case "fmt ":
			if sz < 16 {
				return nil, fmt.Errorf("wav: bad fmt size")
			}
			f := make([]byte, sz)
			if _, err := io.ReadFull(r, f); err != nil {
				return nil, err
			}
			if err := binary.Read(bytes.NewBuffer(f), binary.LittleEndian, &w.Header); err != nil {
				return nil, err
			}
			if w.AudioFormat == wavFormatExtensible {
				// WAVE_FORMAT_EXTENSIBLE wraps the real tag in the first two
				// bytes of the 16-byte SubFormat GUID (the remaining 14 bytes
				// are the fixed KSDATAFORMAT_SUBTYPE_LEGACY suffix). ffmpeg
				// emits this container for multi-channel or 24/32-bit PCM.
				if sz < fmtExtensibleMinSize {
					return nil, fmt.Errorf("wav: bad extensible fmt size: %d", sz)
				}
				w.AudioFormat = binary.LittleEndian.Uint16(f[24:26])
			}
			switch w.AudioFormat {
			case wavFormatPCM:
			case wavFormatIEEEFloat:
			default:
				return nil, fmt.Errorf("wav: unknown audio format: %02x", w.AudioFormat)
			}
			if _, err := w.sampleType(); err != nil {
				return nil, err
			}
			hasFmt = true
		case "data":
			if !hasFmt {
				return nil, fmt.Errorf("wav: unexpected fmt chunk")
			}
			w.Samples = int(sz) / int(w.BitsPerSample) * 8
			w.Duration = time.Duration(w.Samples) * time.Second / time.Duration(w.SampleRate) / time.Duration(w.NumChannels)
			w.r = io.LimitReader(r, int64(sz))
			return &w, nil
		default:
			// Chunks are word-aligned: odd sizes carry one padding byte.
			padded := int64(sz) + int64(sz&1)
			if _, err := io.CopyN(io.Discard, r, padded); err != nil {
				return nil, err
			}
		}
	}
}

// sampleType returns the Go slice element type used for the current format.
func (w *Wav) sampleType() (interface{}, error) {
	switch w.AudioFormat {
	case wavFormatPCM:
		switch w.BitsPerSample {
		case 8:
			return []uint8(nil), nil
		case 16:
			return []int16(nil), nil
		case 24, 32:
			return []int32(nil), nil
		default:
			return nil, fmt.Errorf("wav: unknown bits per sample: %v", w.BitsPerSample)
		}
	case wavFormatIEEEFloat:
		switch w.BitsPerSample {
		case 32:
			return []float32(nil), nil
		case 64:
			return []float64(nil), nil
		default:
			return nil, fmt.Errorf("wav: unknown bits per sample: %v", w.BitsPerSample)
		}
	default:
		return nil, fmt.Errorf("wav: unknown audio format")
	}
}

// ReadSamples returns a [n]T, where T is uint8, int16, int32, float32 or
// float64, based on the wav data. 24- and 32-bit PCM both use int32; 24-bit
// values are sign-extended from their packed 3-byte on-disk form. n is the
// number of interleaved samples to return, across all channels.
func (w *Wav) ReadSamples(n int) (interface{}, error) {
	typ, err := w.sampleType()
	if err != nil {
		return nil, err
	}
	switch typ.(type) {
	case []uint8:
		data := make([]uint8, n)
		if err := binary.Read(w.r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case []int16:
		data := make([]int16, n)
		if err := binary.Read(w.r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case []int32:
		data := make([]int32, n)
		if w.BitsPerSample == 24 {
			// 24-bit PCM is packed little-endian, not a native Go width.
			buf := make([]byte, n*3)
			if _, err := io.ReadFull(w.r, buf); err != nil {
				return nil, err
			}
			for i := range data {
				// Sign-extend the packed 24-bit value to 32 bits.
				v := int32(buf[3*i]) | int32(buf[3*i+1])<<8 | int32(buf[3*i+2])<<16
				data[i] = v << 8 >> 8
			}
		} else if err := binary.Read(w.r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case []float32:
		data := make([]float32, n)
		if err := binary.Read(w.r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case []float64:
		data := make([]float64, n)
		if err := binary.Read(w.r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	default:
		return nil, fmt.Errorf("wav: unknown type: %T", typ)
	}
}

// ReadFloats is like ReadSamples, but it converts any underlying data to a
// float32, scaled to the nominal -1.0..1.0 full-scale range.
func (w *Wav) ReadFloats(n int) ([]float32, error) {
	d, err := w.ReadSamples(n)
	if err != nil {
		return nil, err
	}
	var f []float32
	switch d := d.(type) {
	case []uint8:
		f = make([]float32, len(d))
		for i, v := range d {
			f[i] = float32(v) / math.MaxUint8
		}
	case []int16:
		f = make([]float32, len(d))
		for i, v := range d {
			f[i] = (float32(v) - math.MinInt16) / (math.MaxInt16 - math.MinInt16)
		}
	case []int32:
		f = make([]float32, len(d))
		scale := float32(signedScale(w.BitsPerSample))
		for i, v := range d {
			f[i] = float32(v) / scale
		}
	case []float32:
		f = d
	case []float64:
		f = make([]float32, len(d))
		for i, v := range d {
			f[i] = float32(v)
		}
	default:
		return nil, fmt.Errorf("wav: unknown type: %T", d)
	}
	return f, nil
}

// ReadFloat64s is like ReadFloats, but returns float64 values, which is the
// element type consumed directly by the fft and spectral packages
// (fft.FFTReal, spectral.Pwelch). Multi-channel data stays interleaved;
// split it with Deinterleave before per-channel FFT work.
func (w *Wav) ReadFloat64s(n int) ([]float64, error) {
	d, err := w.ReadSamples(n)
	if err != nil {
		return nil, err
	}
	switch d := d.(type) {
	case []uint8:
		f := make([]float64, len(d))
		for i, v := range d {
			f[i] = float64(v) / math.MaxUint8
		}
		return f, nil
	case []int16:
		f := make([]float64, len(d))
		scale := float64(signedScale(w.BitsPerSample))
		for i, v := range d {
			f[i] = float64(v) / scale
		}
		return f, nil
	case []int32:
		f := make([]float64, len(d))
		scale := float64(signedScale(w.BitsPerSample))
		for i, v := range d {
			f[i] = float64(v) / scale
		}
		return f, nil
	case []float32:
		f := make([]float64, len(d))
		for i, v := range d {
			f[i] = float64(v)
		}
		return f, nil
	case []float64:
		return d, nil
	default:
		return nil, fmt.Errorf("wav: unknown type: %T", d)
	}
}

// Frames returns the number of whole multi-channel frames in n interleaved
// samples given the file's channel count.
func (w *Wav) Frames(n int) int {
	return n / int(w.NumChannels)
}

// signedScale returns the positive full-scale magnitude for a signed PCM
// width: 2^(bits-1). Negative full scale maps to -1.0.
func signedScale(bits uint16) int64 {
	return int64(1) << (bits - 1)
}

// Deinterleave splits an interleaved multi-channel slice into one slice per
// channel. Every channel gets len(samples)/NumChannels frames; trailing
// partial frames are dropped. It is the bridge between the on-disk layout
// and per-channel processing such as fft.FFTReal.
func Deinterleave[T ~uint8 | ~int16 | ~int32 | ~float32 | ~float64](samples []T, channels int) [][]T {
	if channels <= 0 {
		panic(fmt.Sprintf("wav: invalid channel count: %d", channels))
	}
	frames := len(samples) / channels
	out := make([][]T, channels)
	for c := range out {
		out[c] = make([]T, frames)
	}
	for i := 0; i < frames; i++ {
		base := i * channels
		for c := 0; c < channels; c++ {
			out[c][i] = samples[base+c]
		}
	}
	return out
}

// Interleave merges per-channel slices into one interleaved slice. Every
// channel must hold the same number of frames.
func Interleave[T ~uint8 | ~int16 | ~int32 | ~float32 | ~float64](channels [][]T) []T {
	if len(channels) == 0 {
		return nil
	}
	frames := len(channels[0])
	for _, ch := range channels {
		if len(ch) != frames {
			panic("wav: channels have unequal frame counts")
		}
	}
	out := make([]T, frames*len(channels))
	for i := 0; i < frames; i++ {
		base := i * len(channels)
		for c, ch := range channels {
			out[base+c] = ch[i]
		}
	}
	return out
}

// Writer writes WAV files. It streams a canonical RIFF/WAVE container (44
// byte fmt + data header) to an io.WriteSeeker, patching the RIFF and data
// chunk sizes on Close. ByteRate and BlockAlign in the supplied Header are
// derived from the channel count, sample rate and sample width.
type Writer struct {
	Header
	w             io.WriteSeeker
	bytesPerFrame int
	dataBytes     int64
	closed        bool
}

// NewWriter writes the WAV header for h to w and returns a Writer. Only
// AudioFormat, NumChannels, SampleRate and BitsPerSample are consulted;
// ByteRate and BlockAlign are computed.
func NewWriter(w io.WriteSeeker, h Header) (*Writer, error) {
	if h.NumChannels == 0 {
		return nil, fmt.Errorf("wav: NumChannels must be greater than zero")
	}
	if h.SampleRate == 0 {
		return nil, fmt.Errorf("wav: SampleRate must be greater than zero")
	}
	switch h.AudioFormat {
	case wavFormatPCM:
		switch h.BitsPerSample {
		case 8, 16, 24, 32:
		default:
			return nil, fmt.Errorf("wav: unsupported PCM bits per sample: %v", h.BitsPerSample)
		}
	case wavFormatIEEEFloat:
		switch h.BitsPerSample {
		case 32, 64:
		default:
			return nil, fmt.Errorf("wav: unsupported float bits per sample: %v", h.BitsPerSample)
		}
	default:
		return nil, fmt.Errorf("wav: unsupported audio format: %02x", h.AudioFormat)
	}
	bytesPerSample := int(h.BitsPerSample) / 8
	h.BlockAlign = uint16(bytesPerSample * int(h.NumChannels))
	h.ByteRate = h.SampleRate * uint32(h.BlockAlign)

	ww := &Writer{
		Header:        h,
		w:             w,
		bytesPerFrame: int(h.NumChannels),
	}
	if err := ww.writeHeader(); err != nil {
		return nil, err
	}
	return ww, nil
}

// writeHeader emits the fixed 44-byte layout with placeholder chunk sizes.
func (w *Writer) writeHeader() error {
	var buf [44]byte
	copy(buf[0:4], "RIFF")
	// buf[4:8] RIFF size, patched on Close.
	copy(buf[8:12], "WAVE")
	copy(buf[12:16], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:20], 16)
	binary.LittleEndian.PutUint16(buf[20:22], w.AudioFormat)
	binary.LittleEndian.PutUint16(buf[22:24], w.NumChannels)
	binary.LittleEndian.PutUint32(buf[24:28], w.SampleRate)
	binary.LittleEndian.PutUint32(buf[28:32], w.ByteRate)
	binary.LittleEndian.PutUint16(buf[32:34], w.BlockAlign)
	binary.LittleEndian.PutUint16(buf[34:36], w.BitsPerSample)
	copy(buf[36:40], "data")
	// buf[40:44] data size, patched on Close.
	_, err := w.w.Write(buf[:])
	return err
}

// WriteSamples appends interleaved samples. The element type must match the
// declared format: []uint8 (8-bit PCM), []int16 (16-bit PCM), []int32
// (24/32-bit PCM), []float32 (32-bit float) or []float64 (64-bit float).
// The slice length must be a whole number of frames.
func (w *Writer) WriteSamples(samples interface{}) error {
	if w.closed {
		return fmt.Errorf("wav: writer is closed")
	}
	n, err := w.writeSamples(samples)
	if err != nil {
		return err
	}
	if n%w.bytesPerFrame != 0 {
		return fmt.Errorf("wav: %d samples do not form whole frames of %d channels", n, w.NumChannels)
	}
	return nil
}

func (w *Writer) writeSamples(samples interface{}) (int, error) {
	switch s := samples.(type) {
	case []uint8:
		if w.AudioFormat != wavFormatPCM || w.BitsPerSample != 8 {
			return 0, w.typeMismatch()
		}
		if err := binary.Write(w.w, binary.LittleEndian, s); err != nil {
			return 0, err
		}
		w.dataBytes += int64(len(s))
		return len(s), nil
	case []int16:
		if w.AudioFormat != wavFormatPCM || w.BitsPerSample != 16 {
			return 0, w.typeMismatch()
		}
		if err := binary.Write(w.w, binary.LittleEndian, s); err != nil {
			return 0, err
		}
		w.dataBytes += int64(len(s)) * 2
		return len(s), nil
	case []int32:
		if w.AudioFormat != wavFormatPCM {
			return 0, w.typeMismatch()
		}
		switch w.BitsPerSample {
		case 24:
			if err := checkInt24Range(s); err != nil {
				return 0, err
			}
			buf := make([]byte, len(s)*3)
			for i, v := range s {
				u := uint32(v)
				buf[3*i] = byte(u)
				buf[3*i+1] = byte(u >> 8)
				buf[3*i+2] = byte(u >> 16)
			}
			if _, err := w.w.Write(buf); err != nil {
				return 0, err
			}
			w.dataBytes += int64(len(s)) * 3
			return len(s), nil
		case 32:
			if err := binary.Write(w.w, binary.LittleEndian, s); err != nil {
				return 0, err
			}
			w.dataBytes += int64(len(s)) * 4
			return len(s), nil
		default:
			return 0, w.typeMismatch()
		}
	case []float32:
		if w.AudioFormat != wavFormatIEEEFloat || w.BitsPerSample != 32 {
			return 0, w.typeMismatch()
		}
		if err := binary.Write(w.w, binary.LittleEndian, s); err != nil {
			return 0, err
		}
		w.dataBytes += int64(len(s)) * 4
		return len(s), nil
	case []float64:
		if w.AudioFormat != wavFormatIEEEFloat || w.BitsPerSample != 64 {
			return 0, w.typeMismatch()
		}
		if err := binary.Write(w.w, binary.LittleEndian, s); err != nil {
			return 0, err
		}
		w.dataBytes += int64(len(s)) * 8
		return len(s), nil
	default:
		return 0, fmt.Errorf("wav: unsupported sample type %T for format %v/%v bits", samples, w.AudioFormat, w.BitsPerSample)
	}
}

func (w *Writer) typeMismatch() error {
	return fmt.Errorf("wav: sample type does not match format %v/%v bits", w.AudioFormat, w.BitsPerSample)
}

func checkInt24Range(s []int32) error {
	const (
		min24 = -1 << 23
		max24 = 1<<23 - 1
	)
	for i, v := range s {
		if v < min24 || v > max24 {
			return fmt.Errorf("wav: sample %d (%d) out of 24-bit range [%d, %d]", i, v, min24, max24)
		}
	}
	return nil
}

// Close patches the RIFF and data chunk sizes and flushes the header. It
// must be called exactly once.
func (w *Writer) Close() error {
	if w.closed {
		return fmt.Errorf("wav: writer is closed")
	}
	w.closed = true
	var scratch [4]byte
	binary.LittleEndian.PutUint32(scratch[:], uint32(w.dataBytes))
	if _, err := w.w.Seek(40, io.SeekStart); err != nil {
		return err
	}
	if _, err := w.w.Write(scratch[:]); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(scratch[:], uint32(36+w.dataBytes))
	if _, err := w.w.Seek(4, io.SeekStart); err != nil {
		return err
	}
	if _, err := w.w.Write(scratch[:]); err != nil {
		return err
	}
	if _, err := w.w.Seek(0, io.SeekEnd); err != nil {
		return err
	}
	return nil
}

// memSeeker is an in-memory io.WriteSeeker used by Encode and tests.
type memSeeker struct {
	buf []byte
	pos int64
}

func (m *memSeeker) Write(p []byte) (int, error) {
	end := m.pos + int64(len(p))
	if end > int64(len(m.buf)) {
		m.buf = append(m.buf, make([]byte, end-int64(len(m.buf)))...)
	}
	copy(m.buf[m.pos:end], p)
	m.pos = end
	return len(p), nil
}

func (m *memSeeker) Seek(offset int64, whence int) (int64, error) {
	var pos int64
	switch whence {
	case io.SeekStart:
		pos = offset
	case io.SeekCurrent:
		pos = m.pos + offset
	case io.SeekEnd:
		pos = int64(len(m.buf)) + offset
	}
	if pos < 0 {
		return 0, fmt.Errorf("wav: negative seek position")
	}
	m.pos = pos
	return pos, nil
}

// Encode writes samples as a complete WAV file to w, which need not be
// seekable. samples follows the same type and interleaving rules as
// Writer.WriteSamples. For streaming to an io.WriteSeeker use NewWriter.
func Encode(w io.Writer, h Header, samples interface{}) error {
	mem := &memSeeker{}
	ww, err := NewWriter(mem, h)
	if err != nil {
		return err
	}
	if _, err := ww.writeSamples(samples); err != nil {
		return err
	}
	if ww.dataBytes%int64(ww.BlockAlign) != 0 {
		return fmt.Errorf("wav: samples do not form whole frames of %d channels", ww.NumChannels)
	}
	if err := ww.Close(); err != nil {
		return err
	}
	_, err = w.Write(mem.buf)
	return err
}
