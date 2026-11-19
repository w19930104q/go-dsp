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
	"math"
	"math/cmplx"
	"testing"

	"github.com/madelynnblue/go-dsp/fft"
	"github.com/madelynnblue/go-dsp/window"
)

var diagWindows = []struct {
	name string
	fn   func(int) []float64
}{
	{"Rectangular", window.Rectangular},
	{"Bartlett", window.Bartlett},
	{"Hamming", window.Hamming},
	{"Hann", window.Hann},
	{"Blackman", window.Blackman},
	{"FlatTop", window.FlatTop},
}

var diagLengths = []int{64, 256, 1024}

// diagTone returns a unit-amplitude tone of n samples at bin k+frac with
// the given phase.
func diagTone(n, k int, frac, phase float64) []float64 {
	x := make([]float64, n)
	f := (float64(k) + frac) / float64(n)
	for i := range x {
		x[i] = math.Cos(2*math.Pi*f*float64(i) + phase)
	}
	return x
}

// diagWindowedPower returns the power spectrum of x scaled by wf.
func diagWindowedPower(x []float64, wf func(int) []float64) []float64 {
	w := wf(len(x))
	xw := make([]float64, len(x))
	for i := range xw {
		xw[i] = x[i] * w[i]
	}
	spec := fft.FFTReal(xw)
	p := make([]float64, len(x)/2+1)
	for j := range p {
		a := cmplx.Abs(spec[j])
		p[j] = a * a
	}
	return p
}

// oracleLeakage is the ground-truth leaked energy fraction: the energy of a
// windowed tone falling more than halfWidth bins away from its true
// (fractional) frequency.
func oracleLeakage(wf func(int) []float64, n, k int, frac, halfWidth float64) float64 {
	p := diagWindowedPower(diagTone(n, k, frac, 0.7), wf)
	center := (float64(k) + frac)
	var total, inside float64
	for j, v := range p {
		total += v
		if math.Abs(float64(j)-center) <= halfWidth {
			inside += v
		}
	}
	return 1 - inside/total
}

// actualAmpErrorDB is the ground-truth amplitude error of a tone placed
// half a bin off-grid: the attenuation of the highest spectral peak
// relative to the true tone amplitude.
func actualAmpErrorDB(wf func(int) []float64, n, k int) float64 {
	w := wf(n)
	var sum float64
	for _, v := range w {
		sum += v
	}
	p := diagWindowedPower(diagTone(n, k, 0.5, 0), wf)
	peak := 0.0
	for _, v := range p {
		if v > peak {
			peak = v
		}
	}
	// The positive-frequency component of a unit cosine has amplitude
	// sum(w)/2 in the windowed DFT.
	return -10 * math.Log10(4*peak/(sum*sum))
}

// spearman returns the rank correlation of a and b.
func spearman(a, b []float64) float64 {
	rank := func(v []float64) []float64 {
		order := make([]int, len(v))
		for i := range order {
			order[i] = i
		}
		for i := 1; i < len(order); i++ {
			for j := i; j > 0 && v[order[j]] < v[order[j-1]]; j-- {
				order[j], order[j-1] = order[j-1], order[j]
			}
		}
		r := make([]float64, len(v))
		for pos, i := range order {
			r[i] = float64(pos)
		}
		return r
	}
	ra, rb := rank(a), rank(b)
	n := float64(len(ra))
	var d2 float64
	for i := range ra {
		d2 += (ra[i] - rb[i]) * (ra[i] - rb[i])
	}
	return 1 - 6*d2/(n*(n*n-1))
}

// TestDiagnoseLeakageConsistent verifies that the reported LeakageRatio
// matches the ground-truth leaked energy fraction (computed with knowledge
// of the true tone frequency) for every window/length combination, and that
// its ordering across windows agrees with the ordering of the actual
// leakage error.
func TestDiagnoseLeakageConsistent(t *testing.T) {
	const frac = 0.37 // off-bin tone
	for _, n := range diagLengths {
		var metric, oracle []float64
		for _, w := range diagWindows {
			x := diagTone(n, n/4, frac, 0.7)
			d := Diagnose(x, 1000, &PwelchOptions{NFFT: n, Window: w.fn})
			halfWidth := math.Round(d.MainLobeWidthBins / 2)
			o := oracleLeakage(w.fn, n, n/4, frac, halfWidth)
			metric = append(metric, d.LeakageRatio)
			oracle = append(oracle, o)

			// The metric must be within a factor of 2 of the true leaked
			// fraction (both span many orders of magnitude).
			if o > 0 {
				r := d.LeakageRatio / o
				if r < 0.5 || r > 2 {
					t.Errorf("NFFT=%d %s: LeakageRatio %e differs from actual leakage %e (ratio %.2f)",
						n, w.name, d.LeakageRatio, o, r)
				}
			}
		}
		if s := spearman(metric, oracle); s != 1 {
			t.Errorf("NFFT=%d: leakage metric ordering inconsistent with actual error ordering (spearman=%.2f)", n, s)
		}
	}
}

// TestDiagnoseScallopingConsistent verifies that ScallopingLossDB upper
// bounds the actual worst-case tone amplitude error and that its ordering
// across windows matches the actual error ordering, for every window/length
// combination.
func TestDiagnoseScallopingConsistent(t *testing.T) {
	for _, n := range diagLengths {
		var metric, actual []float64
		for _, w := range diagWindows {
			x := diagTone(n, n/4, 0.5, 0)
			d := Diagnose(x, 1000, &PwelchOptions{NFFT: n, Window: w.fn})
			a := actualAmpErrorDB(w.fn, n, n/4)
			metric = append(metric, d.ScallopingLossDB)
			actual = append(actual, a)

			// The reported loss must bound the actual amplitude error.
			if d.ScallopingLossDB+0.05 < a {
				t.Errorf("NFFT=%d %s: ScallopingLossDB %.3f does not bound actual amplitude error %.3f dB",
					n, w.name, d.ScallopingLossDB, a)
			}
		}
		if s := spearman(metric, actual); s != 1 {
			t.Errorf("NFFT=%d: scalloping metric ordering inconsistent with actual error ordering (spearman=%.2f)", n, s)
		}
	}
}

// TestDiagnoseSidelobeBound verifies that no spectral bin outside a tone's
// main lobe exceeds the level predicted by SidelobeLevelDB, for every
// window/length combination.
func TestDiagnoseSidelobeBound(t *testing.T) {
	const frac = 0.37
	for _, n := range diagLengths {
		for _, w := range diagWindows {
			x := diagTone(n, n/4, frac, 0.7)
			d := Diagnose(x, 1000, &PwelchOptions{NFFT: n, Window: w.fn})

			wv := w.fn(n)
			var sum float64
			for _, v := range wv {
				sum += v
			}
			p := diagWindowedPower(x, w.fn)
			center := float64(n/4) + frac
			halfWidth := math.Round(d.MainLobeWidthBins / 2)
			worstDB := math.Inf(-1)
			for j, v := range p {
				if math.Abs(float64(j)-center) > halfWidth {
					if db := 10 * math.Log10(v/(sum*sum/4)); db > worstDB {
						worstDB = db
					}
				}
			}
			// 1 dB tolerance for the finite oversampling grid used to
			// measure the sidelobe level.
			if worstDB > d.SidelobeLevelDB+1 {
				t.Errorf("NFFT=%d %s: bin outside main lobe at %.2f dB exceeds SidelobeLevelDB %.2f",
					n, w.name, worstDB, d.SidelobeLevelDB)
			}
		}
	}
}

// resolvable reports whether two equal-amplitude tones sep bins apart
// produce two distinguishable peaks (a dip between them) in the windowed
// spectrum.
func resolvable(wf func(int) []float64, n int, sep float64) bool {
	k := n / 4
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Cos(2*math.Pi*float64(k)/float64(n)*float64(i)) +
			math.Cos(2*math.Pi*(float64(k)+sep)/float64(n)*float64(i)+0.3)
	}
	p := diagWindowedPower(x, wf)
	lo, hi := k, k+int(math.Ceil(sep))
	p1, p2 := -1, -1
	for j := lo; j <= hi; j++ {
		if p1 < 0 || p[j] > p[p1] {
			p2 = p1
			p1 = j
		} else if p2 < 0 || p[j] > p[p2] {
			p2 = j
		}
	}
	if p1 < 0 || p2 < 0 || p1 == p2 {
		return false
	}
	if p1 > p2 {
		p1, p2 = p2, p1
	}
	dip := p[p1]
	for j := p1; j <= p2; j++ {
		if p[j] < dip {
			dip = p[j]
		}
	}
	// Require at least a 1.5 dB dip between the peaks.
	return dip < 0.7*math.Min(p[p1], p[p2])
}

// TestDiagnoseResolutionConsistent verifies that ResolutionBins tracks the
// actual resolving power: two tones separated by twice ResolutionBins are
// always resolvable, and the ordering of ResolutionBins across windows
// matches the ordering of the actual minimum resolvable separation, for
// every window/length combination.
func TestDiagnoseResolutionConsistent(t *testing.T) {
	for _, n := range diagLengths {
		var metric, minSep []float64
		for _, w := range diagWindows {
			d := Diagnose(diagTone(n, n/4, 0.37, 0), 1000, &PwelchOptions{NFFT: n, Window: w.fn})
			metric = append(metric, d.ResolutionBins)

			ms := math.Inf(1)
			for s := 0.5; s <= 8; s += 0.25 {
				if resolvable(w.fn, n, s) {
					ms = s
					break
				}
			}
			minSep = append(minSep, ms)

			if !resolvable(w.fn, n, 2*d.ResolutionBins) {
				t.Errorf("NFFT=%d %s: tones separated by 2*ResolutionBins (%.2f bins) not resolvable",
					n, w.name, 2*d.ResolutionBins)
			}
		}
		if s := spearman(metric, minSep); s != 1 {
			t.Errorf("NFFT=%d: resolution metric ordering inconsistent with actual resolving power (spearman=%.2f)", n, s)
		}
	}
}

// TestDiagnoseDoesNotChangeResults verifies that computing diagnostics
// neither modifies the input nor affects the results of Pwelch.
func TestDiagnoseDoesNotChangeResults(t *testing.T) {
	n := 256
	x := diagTone(n, n/4, 0.37, 0.7)
	orig := make([]float64, n)
	copy(orig, x)

	o := &PwelchOptions{NFFT: n, Window: window.Hann}
	pxx0, freqs0 := Pwelch(append([]float64(nil), x...), 1000, o)
	_ = Diagnose(x, 1000, o)
	pxx1, freqs1 := Pwelch(append([]float64(nil), x...), 1000, o)

	for i := range x {
		if x[i] != orig[i] {
			t.Fatalf("Diagnose modified input at index %d", i)
		}
	}
	if len(pxx0) != len(pxx1) || len(freqs0) != len(freqs1) {
		t.Fatal("Pwelch result length changed after Diagnose")
	}
	for i := range pxx0 {
		if pxx0[i] != pxx1[i] || freqs0[i] != freqs1[i] {
			t.Fatalf("Pwelch result changed after Diagnose at index %d", i)
		}
	}
}

// TestDiagnoseDefaults verifies default handling and degenerate inputs.
func TestDiagnoseDefaults(t *testing.T) {
	x := diagTone(512, 128, 0.37, 0)

	d := Diagnose(x, 1000, nil)
	if d.NFFT != 256 {
		t.Errorf("default NFFT = %d, want 256", d.NFFT)
	}
	if d.ResolutionHz <= d.BinWidthHz*0.9 {
		t.Errorf("default ResolutionHz %v not above bin width %v", d.ResolutionHz, d.BinWidthHz)
	}
	if d.MaxFreqErrorHz != d.BinWidthHz/2 {
		t.Errorf("MaxFreqErrorHz %v != half bin width %v", d.MaxFreqErrorHz, d.BinWidthHz)
	}
	if d.LeakageRatio < 0 || d.LeakageRatio > 1 {
		t.Errorf("LeakageRatio %v out of [0, 1]", d.LeakageRatio)
	}

	// Empty and short inputs must not panic and must stay in range.
	for _, in := range [][]float64{{}, {1, 2, 3}, make([]float64, 100)} {
		d := Diagnose(in, 1000, &PwelchOptions{NFFT: 256})
		if d.LeakageRatio < 0 || d.LeakageRatio > 1 || math.IsNaN(d.LeakageRatio) {
			t.Errorf("len=%d: LeakageRatio %v out of range", len(in), d.LeakageRatio)
		}
	}

	// Silent input has no leakage.
	d = Diagnose(make([]float64, 512), 1000, nil)
	if d.LeakageRatio != 0 {
		t.Errorf("silent input: LeakageRatio = %v, want 0", d.LeakageRatio)
	}
}
