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

package spectral

import (
	"math/cmplx"

	"github.com/madelynnblue/go-dsp/dsputils"
	"github.com/madelynnblue/go-dsp/fft"
	"github.com/madelynnblue/go-dsp/window"
)

type PwelchOptions struct {
	// NFFT is the number of data points used in each block for the FFT. Must be
	// even; a power 2 is most efficient. This should *NOT* be used to get zero
	// padding, or the scaling of the result will be incorrect. Use Pad for
	// this instead.
	//
	// The default value is 256.
	NFFT int

	// Window is a function that returns an array of window values the length
	// of its input parameter. Each segment is scaled by these values.
	//
	// The default (nil) is window.Hann, from the go-dsp/window package.
	Window func(int) []float64

	// Pad is the number of points to which the data segment is padded when
	// performing the FFT. This can be different from NFFT, which specifies the
	// number of data points used. While not increasing the actual resolution of
	// the psd (the minimum distance between resolvable peaks), this can give
	// more points in the plot, allowing for more detail.
	//
	// The value default is 0, which sets Pad equal to NFFT.
	Pad int

	// Noverlap is the number of points of overlap between blocks.
	//
	// The default value is 0 (no overlap).
	Noverlap int

	// Specifies whether the resulting density values should be scaled by the
	// scaling frequency, which gives density in units of Hz^-1. This allows for
	// integration over the returned frequency values. The default is set for
	// MATLAB compatibility. Note that this is the opposite of matplotlib style,
	// but with equivalent defaults.
	//
	// The default value is false (enable scaling).
	Scale_off bool
}

// Pwelch estimates the power spectral density of x using Welch's method.
// Fs is the sampling frequency (samples per time unit) of x. Fs is used
// to calculate freqs.
// Returns the power spectral density Pxx and corresponding frequencies freqs.
// Designed to be similar to the matplotlib implementation below.
// Reference: http://matplotlib.org/api/mlab_api.html#matplotlib.mlab.psd
// See also: http://www.mathworks.com/help/signal/ref/pwelch.html
func Pwelch(x []float64, Fs float64, o *PwelchOptions) (Pxx, freqs []float64) {
	if len(x) == 0 {
		return []float64{}, []float64{}
	}

	nfft := o.NFFT
	pad := o.Pad
	noverlap := o.Noverlap
	wf := o.Window
	enable_scaling := !o.Scale_off

	if nfft == 0 {
		nfft = 256
	}

	if wf == nil {
		wf = window.Hann
	}

	if pad == 0 {
		pad = nfft
	}

	if len(x) < nfft {
		x = dsputils.ZeroPadF(x, nfft)
	}

	lp := pad/2 + 1
	var scale float64 = 2

	// wlen is the length of each segment after zero padding, matching
	// ZeroPadF(seg, pad) for a segment of length nfft.
	wlen := pad
	if nfft > wlen {
		wlen = nfft
	}

	// The window values depend only on the segment length, so compute them
	// once instead of once per segment.
	wvals := wf(wlen)

	// Segment the input the same way Segment does, but without copying each
	// segment: every segment is copied into a single reused scratch buffer
	// below. This keeps allocation bounded for long signals.
	stride := nfft - noverlap
	var segs int
	if len(x) == nfft {
		segs = 1
	} else if len(x) > nfft {
		segs = (len(x)-nfft)/stride + 1
	}

	Pxx = make([]float64, lp)
	buf := make([]float64, wlen)
	for i := 0; i < segs; i++ {
		seg := x[i*stride : i*stride+nfft]

		// Zero-pad the segment into the scratch buffer, clearing any stale
		// data left over from the previous segment.
		copy(buf, seg)
		clear(buf[len(seg):])

		for j, w := range wvals {
			buf[j] *= w
		}

		pgram := fft.FFTReal(buf)

		for j := range Pxx {
			d := real(cmplx.Conj(pgram[j])*pgram[j]) / float64(segs)

			if j > 0 && j < lp-1 {
				d *= scale
			}

			Pxx[j] += d
		}
	}

	w := wvals
	if nfft != wlen {
		w = wf(nfft)
	}
	var norm float64
	for _, x := range w {
		norm += x * x
	}

	if enable_scaling {
		norm *= Fs
	}

	for i := range Pxx {
		Pxx[i] /= norm
	}

	freqs = make([]float64, lp)
	coef := Fs / float64(pad)
	for i := range freqs {
		freqs[i] = float64(i) * coef
	}

	return
}
