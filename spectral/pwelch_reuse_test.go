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
	"math/rand"
	"testing"

	"github.com/madelynnblue/go-dsp/dsputils"
	"github.com/madelynnblue/go-dsp/fft"
	"github.com/madelynnblue/go-dsp/window"
)

// pwelchReference is the pre-buffer-reuse implementation of Pwelch, kept
// here verbatim as a reference: it allocates fresh buffers for every
// segment. Pwelch must produce bit-identical output.
func pwelchReference(x []float64, Fs float64, o *PwelchOptions) (Pxx, freqs []float64) {
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

	segs := Segment(x, nfft, noverlap)

	Pxx = make([]float64, lp)
	for _, x := range segs {
		x = dsputils.ZeroPadF(x, pad)
		window.Apply(x, wf)

		pgram := fft.FFTReal(x)

		for j := range Pxx {
			d := real(cmplx.Conj(pgram[j])*pgram[j]) / float64(len(segs))

			if j > 0 && j < lp-1 {
				d *= scale
			}

			Pxx[j] += d
		}
	}

	w := wf(nfft)
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

// equalFloat64F reports whether a and b are bit-identical.
func equalFloat64F(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestPwelchBufferReuseIdentical verifies that the buffer-reusing Pwelch is
// bit-identical to the fresh-buffer reference implementation across a range
// of options, and that repeated calls are stable.
func TestPwelchBufferReuseIdentical(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	x := make([]float64, 1<<15)
	for i := range x {
		x[i] = r.NormFloat64()
	}

	opts := []*PwelchOptions{
		{},
		{NFFT: 128},
		{NFFT: 256, Noverlap: 128},
		{NFFT: 256, Pad: 1024},
		{NFFT: 100, Pad: 256, Noverlap: 10},
		{NFFT: 512, Scale_off: true},
		{NFFT: 256, Window: window.Hamming},
	}

	for i, o := range opts {
		wantP, wantF := pwelchReference(x, 44100, o)
		for run := 0; run < 3; run++ {
			gotP, gotF := Pwelch(x, 44100, o)
			if !equalFloat64F(gotP, wantP) {
				t.Fatalf("opts %d run %d: Pxx differs from reference", i, run)
			}
			if !equalFloat64F(gotF, wantF) {
				t.Fatalf("opts %d run %d: freqs differs from reference", i, run)
			}
		}
	}
}

// sinkPxx and sinkFreqs keep results alive so the compiler cannot optimize
// calls away.
var sinkPxx, sinkFreqs []float64

// TestPwelchBufferReuseAllocs verifies that Pwelch on a long signal
// allocates substantially less than the fresh-buffer reference.
func TestPwelchBufferReuseAllocs(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	x := make([]float64, 1<<16)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	o := &PwelchOptions{}

	// Warm the fft buffer pool and factor caches.
	Pwelch(x, 44100, o)
	pwelchReference(x, 44100, o)

	ref := testing.AllocsPerRun(20, func() { sinkPxx, sinkFreqs = pwelchReference(x, 44100, o) })
	got := testing.AllocsPerRun(20, func() { sinkPxx, sinkFreqs = Pwelch(x, 44100, o) })

	if got >= ref {
		t.Errorf("Pwelch allocs/run = %v, want less than reference %v", got, ref)
	}
}
