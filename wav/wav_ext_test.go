package wav_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/madelynnblue/go-dsp/fft"
	"github.com/madelynnblue/go-dsp/wav"
)

const (
	roundChannels = 4
	roundFrames   = 6
	roundSamples  = roundChannels * roundFrames
	roundRate     = 48000
)

// Fixed, hand-picked multi-channel fixtures. Every width exercises its
// full-scale extremes, zero and multi-byte values; values repeat neither
// per channel nor per frame so interleaving order errors cannot hide.
func fixedSamples() map[string]interface{} {
	return map[string]interface{}{
		"pcm8": []uint8{
			0, 255, 128, 127,
			1, 254, 130, 126,
			100, 200, 50, 205,
			7, 248, 64, 192,
			33, 222, 99, 156,
			12, 243, 77, 178,
		},
		"pcm16": []int16{
			math.MinInt16, math.MaxInt16, 0, -1,
			1, -2, 19530, -19531,
			30000, -30001, 255, -256,
			12345, -12346, 1000, -1001,
			130, -131, 16383, -16384,
			4096, -4097, 3333, -3334,
		},
		"pcm24": []int32{
			-8388608, 8388607, 0, -1,
			1, -2, 195303, -195304,
			7000000, -7000001, 8388606, -8388607,
			255, -256, 65535, -65536,
			130, -131, 5000000, -5000000,
			1234567, -7654321, 333333, -333334,
		},
		"pcm32": []int32{
			math.MinInt32, math.MaxInt32, 0, -1,
			1, -2, 195303, -195304,
			7000000, -7000001, 8388607, -8388608,
			1073741823, -1073741824, 65535, -65536,
			130, -131, 5000000, -5000000,
			1234567, -7654321, 333333, -333334,
		},
		"float32": []float32{
			0, 1, -1, 0.5,
			-0.5, 0.25, -0.125, 3.1415927,
			-3.1415927, 1e-3, -1e-3, 100.5,
			-100.5, 0.0078125, -0.0078125, 2,
			-2, 0.999, -0.999, 16,
			-16, 1e-6, -1e-6, 42,
		},
		"float64": []float64{
			0, 1, -1, 0.5,
			-0.5, 0.25, -0.125, math.Pi,
			-math.Pi, 1e-3, -1e-3, 100.5,
			-100.5, 0.0078125, -0.0078125, 2,
			-2, 0.999, -0.999, 16,
			-16, 1e-12, -1e-12, 42,
		},
	}
}

func roundTripHeader(name string) wav.Header {
	h := wav.Header{
		NumChannels: roundChannels,
		SampleRate:  roundRate,
	}
	switch name {
	case "pcm8":
		h.AudioFormat, h.BitsPerSample = 1, 8
	case "pcm16":
		h.AudioFormat, h.BitsPerSample = 1, 16
	case "pcm24":
		h.AudioFormat, h.BitsPerSample = 1, 24
	case "pcm32":
		h.AudioFormat, h.BitsPerSample = 1, 32
	case "float32":
		h.AudioFormat, h.BitsPerSample = 3, 32
	case "float64":
		h.AudioFormat, h.BitsPerSample = 3, 64
	}
	return h
}

// seekBuffer is a test-local in-memory io.WriteSeeker for the streaming API.
type seekBuffer struct {
	buf bytes.Buffer
	pos int64
}

func (s *seekBuffer) Write(p []byte) (int, error) {
	end := s.pos + int64(len(p))
	if end > int64(s.buf.Len()) {
		s.buf.Grow(int(end) - s.buf.Len())
		s.buf.Write(make([]byte, int(end)-s.buf.Len()))
	}
	copy(s.buf.Bytes()[s.pos:], p)
	s.pos = end
	return len(p), nil
}

func (s *seekBuffer) Seek(offset int64, whence int) (int64, error) {
	var pos int64
	switch whence {
	case 0:
		pos = offset
	case 1:
		pos = s.pos + offset
	case 2:
		pos = int64(s.buf.Len()) + offset
	}
	s.pos = pos
	return pos, nil
}

func TestEncodeDecodeRoundTripFixedMultichannel(t *testing.T) {
	for name, samples := range fixedSamples() {
		t.Run(name, func(t *testing.T) {
			h := roundTripHeader(name)
			var buf bytes.Buffer
			if err := wav.Encode(&buf, h, samples); err != nil {
				t.Fatalf("Encode: %v", err)
			}

			r, err := wav.New(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if r.AudioFormat != h.AudioFormat || r.BitsPerSample != h.BitsPerSample ||
				r.NumChannels != roundChannels || r.SampleRate != roundRate {
				t.Fatalf("header mismatch: got %+v", r.Header)
			}
			if r.Samples != roundSamples {
				t.Fatalf("Samples = %d, want %d", r.Samples, roundSamples)
			}
			if r.Frames(r.Samples) != roundFrames {
				t.Fatalf("Frames = %d, want %d", r.Frames(r.Samples), roundFrames)
			}
			wantBlockAlign := int(h.BitsPerSample) / 8 * roundChannels
			if int(r.BlockAlign) != wantBlockAlign ||
				int(r.ByteRate) != roundRate*wantBlockAlign {
				t.Fatalf("block layout mismatch: align %d rate %d", r.BlockAlign, r.ByteRate)
			}

			got, err := r.ReadSamples(roundSamples)
			if err != nil {
				t.Fatalf("ReadSamples: %v", err)
			}
			if !reflect.DeepEqual(got, samples) {
				t.Fatalf("sample values not strictly preserved\ngot  %v\nwant %v", got, samples)
			}

			// Interleaving order must be channel-major within each frame.
			chans := deinterleaveAny(t, got, roundChannels)
			want := reflect.ValueOf(samples)
			for c := 0; c < roundChannels; c++ {
				for i := 0; i < roundFrames; i++ {
					if gotVal := reflect.ValueOf(chans[c]).Index(i); !gotVal.Equal(want.Index(i*roundChannels + c)) {
						t.Fatalf("frame %d channel %d misplaced: got %v want %v",
							i, c, gotVal.Interface(), want.Index(i*roundChannels+c).Interface())
					}
				}
			}
		})
	}
}

func deinterleaveAny(t *testing.T, samples interface{}, channels int) []interface{} {
	t.Helper()
	switch s := samples.(type) {
	case []uint8:
		d := wav.Deinterleave(s, channels)
		out := make([]interface{}, len(d))
		for i := range d {
			out[i] = d[i]
		}
		return out
	case []int16:
		d := wav.Deinterleave(s, channels)
		out := make([]interface{}, len(d))
		for i := range d {
			out[i] = d[i]
		}
		return out
	case []int32:
		d := wav.Deinterleave(s, channels)
		out := make([]interface{}, len(d))
		for i := range d {
			out[i] = d[i]
		}
		return out
	case []float32:
		d := wav.Deinterleave(s, channels)
		out := make([]interface{}, len(d))
		for i := range d {
			out[i] = d[i]
		}
		return out
	case []float64:
		d := wav.Deinterleave(s, channels)
		out := make([]interface{}, len(d))
		for i := range d {
			out[i] = d[i]
		}
		return out
	default:
		t.Fatalf("unsupported type %T", samples)
		return nil
	}
}

func TestStreamingWriterMatchesEncode(t *testing.T) {
	h := roundTripHeader("pcm24")
	samples := fixedSamples()["pcm24"].([]int32)

	var encoded bytes.Buffer
	if err := wav.Encode(&encoded, h, samples); err != nil {
		t.Fatal(err)
	}

	sb := &seekBuffer{}
	ww, err := wav.NewWriter(sb, h)
	if err != nil {
		t.Fatal(err)
	}
	// Two frame-aligned writes exercise the streaming path.
	if err := ww.WriteSamples(samples[:roundChannels*2]); err != nil {
		t.Fatal(err)
	}
	if err := ww.WriteSamples(samples[roundChannels*2:]); err != nil {
		t.Fatal(err)
	}
	if err := ww.Close(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(sb.buf.Bytes(), encoded.Bytes()) {
		t.Fatal("streaming Writer and Encode produced different bytes")
	}
	if err := ww.Close(); err == nil {
		t.Fatal("double Close should error")
	}
}

// The 16-bit path must stay byte-for-byte compatible with the legacy
// behavior: the existing fixture still reads the same int16 samples, and
// canonical 16-bit files written by the new writer are deterministic and
// survive a write/read round trip with identical payload bytes.
func Test16BitPathByteIdentical(t *testing.T) {
	raw, err := os.ReadFile("small.wav")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("small.wav")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r, err := wav.New(f)
	if err != nil {
		t.Fatal(err)
	}
	original, err := r.ReadSamples(r.Samples)
	if err != nil {
		t.Fatal(err)
	}
	originalSamples := original.([]int16)

	// The legacy fixture carries oversized RIFF/data size fields; reading
	// only touches the declared sample region, whose bytes must match.
	legacyPayload := raw[44 : 44+r.Samples*2]
	legacyWire := make([]byte, r.Samples*2)
	for i, v := range originalSamples {
		binary.LittleEndian.PutUint16(legacyWire[2*i:], uint16(v))
	}
	if !bytes.Equal(legacyPayload, legacyWire) {
		t.Fatal("16-bit int16 samples no longer match the fixture payload bytes")
	}

	// Re-encoding canonicalizes the chunk sizes; the declared sample
	// payload itself must be byte-for-byte identical to the fixture's.
	var out bytes.Buffer
	if err := wav.Encode(&out, r.Header, originalSamples); err != nil {
		t.Fatal(err)
	}
	enc := out.Bytes()
	if !bytes.Equal(enc[44:], legacyPayload) {
		t.Fatal("re-encoded 16-bit payload bytes differ from fixture")
	}

	// Canonical header fields, including exact RIFF/data sizes.
	if string(enc[:4]) != "RIFF" || string(enc[8:12]) != "WAVE" ||
		string(enc[12:16]) != "fmt " || string(enc[36:40]) != "data" {
		t.Fatal("unexpected container layout")
	}
	dataSize := binary.LittleEndian.Uint32(enc[40:44])
	if int(dataSize) != r.Samples*2 {
		t.Fatalf("data size = %d, want %d", dataSize, r.Samples*2)
	}
	if riffSize := binary.LittleEndian.Uint32(enc[4:8]); int(riffSize) != 36+len(legacyPayload) {
		t.Fatalf("riff size = %d, want %d", riffSize, 36+len(legacyPayload))
	}

	// Deterministic: a second encode produces the exact same file.
	var out2 bytes.Buffer
	if err := wav.Encode(&out2, r.Header, originalSamples); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out2.Bytes(), enc) {
		t.Fatal("16-bit encoding is not deterministic")
	}

	// Re-reading gives back exactly the same int16 samples.
	r2, err := wav.New(bytes.NewReader(enc))
	if err != nil {
		t.Fatal(err)
	}
	again, err := r2.ReadSamples(r2.Samples)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(again, original) {
		t.Fatal("16-bit samples changed across re-encode")
	}
}

func TestPacked24BitWireBytes(t *testing.T) {
	// The on-disk form must be packed 3-byte little endian, sign extended
	// only when read back into int32.
	h := roundTripHeader("pcm24")
	samples := fixedSamples()["pcm24"].([]int32)
	var buf bytes.Buffer
	if err := wav.Encode(&buf, h, samples); err != nil {
		t.Fatal(err)
	}
	payload := buf.Bytes()[44:]
	if len(payload) != roundSamples*3 {
		t.Fatalf("payload size = %d, want %d", len(payload), roundSamples*3)
	}
	for i, v := range samples {
		u := uint32(v)
		want := [3]byte{byte(u), byte(u >> 8), byte(u >> 16)}
		if got := payload[3*i : 3*i+3]; !bytes.Equal(got, want[:]) {
			t.Fatalf("sample %d wire bytes = %x, want %x", i, got, want)
		}
	}
}

func TestFloatScaling(t *testing.T) {
	cases := []struct {
		name    string
		samples interface{}
		wantF64 []float64
		relTol  float64
	}{
		{"pcm24", fixedSamples()["pcm24"].([]int32), nil, 0},
		{"pcm32", fixedSamples()["pcm32"].([]int32), nil, 0},
		{"float32", fixedSamples()["float32"].([]float32), nil, 1e-7},
		{"float64", fixedSamples()["float64"].([]float64), nil, 0},
		{"pcm8", fixedSamples()["pcm8"].([]uint8), nil, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := roundTripHeader(tc.name)
			var buf bytes.Buffer
			if err := wav.Encode(&buf, h, tc.samples); err != nil {
				t.Fatal(err)
			}
			r64, err := wav.New(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t.Fatal(err)
			}
			got64, err := r64.ReadFloat64s(roundSamples)
			if err != nil {
				t.Fatal(err)
			}
			r32, err := wav.New(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t.Fatal(err)
			}
			got32, err := r32.ReadFloats(roundSamples)
			if err != nil {
				t.Fatal(err)
			}
			for i := range got64 {
				var want float64
				switch s := tc.samples.(type) {
				case []int32:
					bits := h.BitsPerSample
					want = float64(s[i]) / float64(int64(1)<<(bits-1))
				case []float32:
					want = float64(s[i])
				case []float64:
					want = s[i]
				case []uint8:
					want = float64(s[i]) / math.MaxUint8
				}
				if math.Abs(got64[i]-want) > tc.relTol+1e-12 {
					t.Fatalf("ReadFloat64s[%d] = %v, want %v", i, got64[i], want)
				}
				if math.Abs(float64(got32[i])-want) > 1e-6 {
					t.Fatalf("ReadFloats[%d] = %v, want %v", i, got32[i], want)
				}
			}
		})
	}

	// Exact full-scale anchors for the new widths.
	anchors := []struct {
		name    string
		bits    uint16
		samples []int32
	}{
		{"pcm24", 24, []int32{-8388608, 8388607, 0}},
		{"pcm32", 32, []int32{math.MinInt32, math.MaxInt32, 0}},
	}
	for _, a := range anchors {
		h := wav.Header{AudioFormat: 1, BitsPerSample: a.bits, NumChannels: 1, SampleRate: 8000}
		var buf bytes.Buffer
		if err := wav.Encode(&buf, h, a.samples); err != nil {
			t.Fatal(err)
		}
		r, err := wav.New(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatal(err)
		}
		got, err := r.ReadFloat64s(len(a.samples))
		if err != nil {
			t.Fatal(err)
		}
		if got[0] != -1.0 || got[2] != 0.0 {
			t.Fatalf("%s full-scale anchors wrong: %v", a.name, got)
		}
	}
}

func TestInterleaveDeinterleave(t *testing.T) {
	samples := fixedSamples()["pcm24"].([]int32)
	chans := wav.Deinterleave(samples, roundChannels)
	if len(chans) != roundChannels {
		t.Fatalf("got %d channels", len(chans))
	}
	for c, ch := range chans {
		if len(ch) != roundFrames {
			t.Fatalf("channel %d length = %d", c, len(ch))
		}
		if ch[0] != samples[c] {
			t.Fatalf("channel %d first frame wrong: %d want %d", c, ch[0], samples[c])
		}
	}
	if back := wav.Interleave(chans); !reflect.DeepEqual(back, samples) {
		t.Fatal("Interleave(Deinterleave(x)) != x")
	}

	// A trailing partial frame is dropped, not scrambled.
	partial := wav.Deinterleave(append(append([]int32{}, samples...), 42), roundChannels)
	for _, ch := range partial {
		if len(ch) != roundFrames {
			t.Fatalf("partial frame not dropped: len %d", len(ch))
		}
	}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Deinterleave with zero channels should panic")
			}
		}()
		_ = wav.Deinterleave(samples, 0)
	}()
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Interleave of unequal channels should panic")
			}
		}()
		_ = wav.Interleave([][]int16{{1, 2}, {3}})
	}()
	if got := wav.Interleave([][]uint8{}); got != nil {
		t.Fatalf("Interleave(empty) = %v, want nil", got)
	}
}

func TestWriterValidation(t *testing.T) {
	good := wav.Header{AudioFormat: 1, BitsPerSample: 24, NumChannels: 2, SampleRate: 48000}
	badHeaders := []wav.Header{
		{AudioFormat: 1, BitsPerSample: 12, NumChannels: 2, SampleRate: 48000},
		{AudioFormat: 3, BitsPerSample: 16, NumChannels: 2, SampleRate: 48000},
		{AudioFormat: 9, BitsPerSample: 16, NumChannels: 2, SampleRate: 48000},
		{AudioFormat: 1, BitsPerSample: 24, NumChannels: 0, SampleRate: 48000},
		{AudioFormat: 1, BitsPerSample: 24, NumChannels: 2, SampleRate: 0},
	}
	for i, h := range badHeaders {
		if _, err := wav.NewWriter(&seekBuffer{}, h); err == nil {
			t.Fatalf("bad header %d accepted: %+v", i, h)
		}
	}

	ww, err := wav.NewWriter(&seekBuffer{}, good)
	if err != nil {
		t.Fatal(err)
	}
	if err := ww.WriteSamples([]int16{1, 2}); err == nil {
		t.Fatal("type mismatch accepted")
	}
	if err := ww.WriteSamples([]int32{1 << 23}); err == nil {
		t.Fatal("out-of-range 24-bit value accepted")
	}
	if err := ww.WriteSamples([]int32{1, 2, 3}); err == nil {
		t.Fatal("non-frame-aligned samples accepted")
	}
	if err := ww.WriteSamples([]int32{1, 2}); err != nil {
		t.Fatalf("valid write failed: %v", err)
	}
	if err := ww.Close(); err != nil {
		t.Fatal(err)
	}
	if err := ww.WriteSamples([]int32{1, 2}); err == nil {
		t.Fatal("write after Close accepted")
	}

	// Encode shares the same checks.
	if err := wav.Encode(&bytes.Buffer{}, good, []int32{1, 2, 3}); err == nil {
		t.Fatal("Encode accepted non-frame-aligned samples")
	}
	if err := wav.Encode(&bytes.Buffer{}, good, []int16{1, 2}); err == nil {
		t.Fatal("Encode accepted mismatched type")
	}
}

func TestRejectsUnsupportedReadWidth(t *testing.T) {
	// Hand-built PCM/12-bit mono file: valid container, unsupported width.
	var buf bytes.Buffer
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, uint32(40))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))
	binary.Write(&buf, binary.LittleEndian, wav.Header{
		AudioFormat: 1, NumChannels: 1, SampleRate: 8000,
		ByteRate: 12000, BlockAlign: 1, BitsPerSample: 12,
	})
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(0))
	if _, err := wav.New(bytes.NewReader(buf.Bytes())); err == nil {
		t.Fatal("unsupported 12-bit PCM should be rejected at New")
	}
}

func TestUnknownChunksSkippedWithPadding(t *testing.T) {
	// A JUNK chunk with an odd (padded) size must be skipped correctly.
	samples := fixedSamples()["pcm16"].([]int16)
	var plain bytes.Buffer
	if err := wav.Encode(&plain, roundTripHeader("pcm16"), samples); err != nil {
		t.Fatal(err)
	}
	var padded bytes.Buffer
	padded.Write(plain.Bytes()[:12]) // RIFF ... WAVE
	padded.WriteString("JUNK")
	binary.Write(&padded, binary.LittleEndian, uint32(3)) // odd size -> 4 bytes on disk
	padded.Write([]byte{0xaa, 0xbb, 0xcc, 0x00})
	padded.Write(plain.Bytes()[12:]) // fmt + data verbatim

	r, err := wav.New(bytes.NewReader(padded.Bytes()))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := r.ReadSamples(roundSamples)
	if err != nil {
		t.Fatalf("ReadSamples: %v", err)
	}
	if !reflect.DeepEqual(got, samples) {
		t.Fatal("samples corrupted by padded unknown chunk")
	}
}

// ---- ffmpeg interop -------------------------------------------------------

func ffmpegAvailable(t *testing.T) string {
	t.Helper()
	bin, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not available")
	}
	return bin
}

func ffmpegRaw(t *testing.T, bin, wavPath, rawFormat string) []byte {
	t.Helper()
	cmd := exec.Command(bin, "-v", "error", "-i", wavPath, "-f", rawFormat, "-")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("ffmpeg decode %s: %v\n%s", rawFormat, err, stderr.String())
	}
	return out
}

func ffmpegProbeChannels(t *testing.T, bin, wavPath string) string {
	t.Helper()
	cmd := exec.Command(bin, "-i", wavPath, "-f", "null", "-")
	out, _ := cmd.CombinedOutput()
	return string(out)
}

// Every file we write must be decodable by ffmpeg, and the decoded samples
// must numerically match what we put in (standard full-scale normalization).
func TestFFmpegDecodesWrittenFiles(t *testing.T) {
	bin := ffmpegAvailable(t)
	dir := t.TempDir()

	rawFormats := map[string]string{
		"pcm8":    "u8",
		"pcm16":   "s16le",
		"pcm24":   "s24le",
		"pcm32":   "s32le",
		"float32": "f32le",
		"float64": "f64le",
	}
	for name, samples := range fixedSamples() {
		t.Run(name, func(t *testing.T) {
			h := roundTripHeader(name)
			var buf bytes.Buffer
			if err := wav.Encode(&buf, h, samples); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, name+".wav")
			if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
				t.Fatal(err)
			}

			probe := ffmpegProbeChannels(t, bin, path)
			// ffmpeg reports either "4 channels" or the guessed layout "4.0".
			if !strings.Contains(probe, "4 channels") && !strings.Contains(probe, " 4.0,") {
				t.Fatalf("ffmpeg does not see 4 channels:\n%s", probe)
			}

			// Native raw decode: payload bytes must survive untouched.
			raw := ffmpegRaw(t, bin, path, rawFormats[name])
			if !bytes.Equal(raw, buf.Bytes()[44:]) {
				t.Fatalf("ffmpeg raw %s output does not match written payload", rawFormats[name])
			}

			// Float decode: numeric values must match standard normalization.
			f32raw := ffmpegRaw(t, bin, path, "f32le")
			decoded := make([]float32, len(f32raw)/4)
			if err := binary.Read(bytes.NewReader(f32raw), binary.LittleEndian, decoded); err != nil {
				t.Fatal(err)
			}
			if len(decoded) != roundSamples {
				t.Fatalf("decoded %d samples, want %d", len(decoded), roundSamples)
			}
			for i := 0; i < roundSamples; i++ {
				var want float64
				switch s := samples.(type) {
				case []uint8:
					// ffmpeg treats 8-bit PCM as offset-binary centered at 128.
					want = float64(int(s[i])-128) / 128
				case []int16:
					want = float64(s[i]) / 32768
				case []int32:
					want = float64(s[i]) / float64(int64(1)<<(h.BitsPerSample-1))
				case []float32:
					want = float64(s[i])
				case []float64:
					want = s[i]
				}
				if math.Abs(float64(decoded[i])-want) > 1e-6 {
					t.Fatalf("sample %d: ffmpeg decoded %v, want %v", i, decoded[i], want)
				}
			}
		})
	}
}

// Files produced by ffmpeg itself (24/32-bit PCM and 64-bit float, stereo)
// must read back correctly, sample-for-sample against ffmpeg's own raw decode.
func TestReadsFFmpegEncodedFiles(t *testing.T) {
	bin := ffmpegAvailable(t)
	dir := t.TempDir()
	cases := []struct {
		name   string
		codec  string
		rawFmt string
		bits   uint16
	}{
		{"ext24", "pcm_s24le", "s24le", 24},
		{"ext32", "pcm_s32le", "s32le", 32},
		{"extf64", "pcm_f64le", "f64le", 64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".wav")
			cmd := exec.Command(bin, "-y", "-v", "error",
				"-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.02",
				"-ac", "2", "-c:a", tc.codec, path)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("ffmpeg encode: %v\n%s", err, out)
			}

			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			r, err := wav.New(f)
			if err != nil {
				t.Fatal(err)
			}
			if r.NumChannels != 2 {
				t.Fatalf("channels = %d, want 2", r.NumChannels)
			}
			if tc.bits <= 32 {
				if r.AudioFormat != 1 || r.BitsPerSample != tc.bits {
					t.Fatalf("format = %v/%v bits", r.AudioFormat, r.BitsPerSample)
				}
			} else if r.AudioFormat != 3 || r.BitsPerSample != 64 {
				t.Fatalf("format = %v/%v bits", r.AudioFormat, r.BitsPerSample)
			}

			got, err := r.ReadSamples(r.Samples)
			if err != nil {
				t.Fatal(err)
			}
			raw := ffmpegRaw(t, bin, path, tc.rawFmt)
			if len(raw) != r.Samples*(int(tc.bits)/8) {
				t.Fatalf("raw size %d does not cover %d samples", len(raw), r.Samples)
			}
			switch g := got.(type) {
			case []int32:
				width := int(tc.bits) / 8
				for i, v := range g {
					var want int32
					for b := 0; b < width; b++ {
						want |= int32(raw[i*width+b]) << (8 * b)
					}
					if tc.bits == 24 {
						want <<= 8
						want >>= 8 // sign extend
					}
					if v != want {
						t.Fatalf("sample %d = %d, ffmpeg raw says %d", i, v, want)
					}
				}
			case []float64:
				for i, v := range g {
					want := math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
					if v != want {
						t.Fatalf("sample %d = %v, ffmpeg raw says %v", i, v, want)
					}
				}
			default:
				t.Fatalf("unexpected read type %T", got)
			}
		})
	}
}

// WAV data must flow straight into the FFT toolchain and back.
func TestFFTInterop(t *testing.T) {
	h := roundTripHeader("pcm24")
	samples := fixedSamples()["pcm24"].([]int32)
	var buf bytes.Buffer
	if err := wav.Encode(&buf, h, samples); err != nil {
		t.Fatal(err)
	}
	r, err := wav.New(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	floats, err := r.ReadFloat64s(roundSamples)
	if err != nil {
		t.Fatal(err)
	}
	channels := wav.Deinterleave(floats, roundChannels)

	for c, ch := range channels {
		spectrum := fft.FFTReal(ch)
		back := fft.IFFT(spectrum)
		for i, z := range back {
			if math.Abs(real(z)-ch[i]) > 1e-12 || math.Abs(imag(z)) > 1e-12 {
				t.Fatalf("channel %d FFT roundtrip failed at %d: %v vs %v", c, i, z, ch[i])
			}
		}
	}

	// Processed float64 data goes back through the writer and survives.
	processed := make([][]float64, len(channels))
	for c, ch := range channels {
		processed[c] = append([]float64(nil), ch...)
		for i := range processed[c] {
			processed[c][i] *= 0.25
		}
	}
	interleaved := wav.Interleave(processed)
	var fbuf bytes.Buffer
	fh := wav.Header{
		AudioFormat: 3, BitsPerSample: 64,
		NumChannels: roundChannels, SampleRate: roundRate,
	}
	if err := wav.Encode(&fbuf, fh, interleaved); err != nil {
		t.Fatal(err)
	}
	r2, err := wav.New(bytes.NewReader(fbuf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	again, err := r2.ReadSamples(roundSamples)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(again, interleaved) {
		t.Fatal("float64 samples changed through write/read FFT interop path")
	}
}
