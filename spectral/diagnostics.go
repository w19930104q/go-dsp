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

	"github.com/madelynnblue/go-dsp/dsputils"
	"github.com/madelynnblue/go-dsp/fft"
	"github.com/madelynnblue/go-dsp/window"
)

// Diagnostics describes confidence metrics for a spectral analysis
// configuration (window function, segment length, padding and sample rate)
// applied to a given input. All window-dependent values are measured from
// the actual window function's discrete-time Fourier transform, so they are
// valid for any window/length combination, including user-supplied windows.
//
// The metrics are designed so that their ordering matches the ordering of
// the actual analysis error: configurations reporting worse leakage or
// resolution metrics produce correspondingly worse measured errors.
//
// Diagnose does not modify x and does not change the results of Pwelch; it
// only reports properties of the configuration and input.
type Diagnostics struct {
	// NFFT is the segment length the metrics apply to (after defaults).
	NFFT int

	// BinWidthHz is the spacing between adjacent FFT bins, Fs / Pad.
	BinWidthHz float64

	// ResolutionBins is the equivalent noise bandwidth of the window in
	// bins: NFFT * sum(w^2) / sum(w)^2. Two equal-power tones must be
	// separated by at least roughly this many bins to be resolved.
	ResolutionBins float64

	// ResolutionHz is ResolutionBins expressed in Hz. It is the minimum
	// frequency separation of two tones that the configuration can
	// reliably distinguish; larger values mean worse resolution.
	ResolutionHz float64

	// MaxFreqErrorHz is the worst-case error when a peak's frequency is
	// read from the nearest FFT bin: half of BinWidthHz.
	MaxFreqErrorHz float64

	// CoherentGain is the mean of the window, sum(w) / NFFT. Peak
	// amplitudes must be divided by this value to recover true tone
	// amplitudes.
	CoherentGain float64

	// MainLobeWidthBins is the measured full width of the window
	// transform's main lobe, in bins.
	MainLobeWidthBins float64

	// SidelobeLevelDB is the measured level of the window's highest
	// sidelobe relative to the main lobe peak, in dB (a negative number).
	// Values closer to 0 mean stronger spectral leakage: energy from a
	// tone contaminates distant bins at up to this relative level.
	SidelobeLevelDB float64

	// ScallopingLossDB is the measured worst-case amplitude error, in dB
	// (a positive number), of a tone falling exactly between two bins.
	// The actual peak amplitude error of any single tone is bounded by
	// this value.
	ScallopingLossDB float64

	// LeakageRatio is an input-dependent leakage indicator in [0, 1]:
	// the fraction of total spectral power of the (windowed, segmented)
	// input that falls outside the main lobes of the detected peaks.
	// 0 means all energy is confined to main lobes (no leakage); values
	// near 1 mean the spectrum is dominated by leaked energy and peak
	// measurements from it should not be trusted.
	LeakageRatio float64
}

// windowOversample is the zero-padding factor used to measure window
// transform properties (sidelobe level, scalloping loss, main lobe width).
const windowOversample = 64

// Diagnose computes confidence metrics for the spectral analysis of x with
// sample rate Fs and options o, using the same defaults as Pwelch. A nil o
// is equivalent to a zero PwelchOptions. It does not modify x and has no
// effect on the results of Pwelch.
func Diagnose(x []float64, Fs float64, o *PwelchOptions) Diagnostics {
	var nfft, pad, noverlap int
	var wf func(int) []float64
	if o != nil {
		nfft = o.NFFT
		pad = o.Pad
		noverlap = o.Noverlap
		wf = o.Window
	}

	if nfft == 0 {
		nfft = 256
	}
	if wf == nil {
		wf = window.Hann
	}
	if pad == 0 {
		pad = nfft
	}

	w := wf(nfft)
	wm := analyzeWindow(w)

	d := Diagnostics{
		NFFT:              nfft,
		BinWidthHz:        Fs / float64(pad),
		ResolutionBins:    wm.enbwBins,
		ResolutionHz:      wm.enbwBins * Fs / float64(nfft),
		MaxFreqErrorHz:    Fs / float64(pad) / 2,
		CoherentGain:      wm.coherentGain,
		MainLobeWidthBins: wm.mainLobeWidthBins,
		SidelobeLevelDB:   wm.sidelobeDB,
		ScallopingLossDB:  wm.scallopingLossDB,
		LeakageRatio:      leakageRatio(x, nfft, pad, noverlap, wf, wm.mainLobeWidthBins, wm.sidelobeDB),
	}

	return d
}

// windowMetrics holds window transform properties measured by analyzeWindow.
type windowMetrics struct {
	enbwBins          float64
	coherentGain      float64
	mainLobeWidthBins float64
	sidelobeDB        float64
	scallopingLossDB  float64
}

// analyzeWindow measures resolution and leakage properties of the window w
// from its (oversampled) discrete-time Fourier transform.
func analyzeWindow(w []float64) windowMetrics {
	n := len(w)
	if n == 0 {
		return windowMetrics{}
	}

	var sum, sumSq float64
	for _, v := range w {
		sum += v
		sumSq += v * v
	}

	m := windowMetrics{
		coherentGain: sum / float64(n),
	}
	if sum != 0 {
		m.enbwBins = float64(n) * sumSq / (sum * sum)
	}

	// Oversampled magnitude spectrum of the window. Zero-padding by
	// windowOversample gives a frequency grid of 1/windowOversample bins.
	L := dsputils.NextPowerOf2(n * windowOversample)
	wp := dsputils.ZeroPadF(append([]float64(nil), w...), L)
	spec := fft.FFTReal(wp)
	half := L/2 + 1
	mag := make([]float64, half)
	peak := 0.0
	peakIdx := 0
	for i := 0; i < half; i++ {
		mag[i] = cmplx.Abs(spec[i])
		if mag[i] > peak {
			peak = mag[i]
			peakIdx = i
		}
	}
	if peak == 0 {
		return m
	}

	// Main lobe: walk from the peak to the first null. A null is a local
	// minimum well below the peak; the depth requirement skips shallow
	// ripples on flat-topped main lobes (e.g. the flat top window, whose
	// transform is not monotone right after the peak). The main lobe is
	// symmetric, so the full width is twice the distance from the peak to
	// the null, in units of bins.
	const nullDepth = 0.05
	nullIdx := -1
	for i := peakIdx + 1; i+1 < half; i++ {
		if mag[i] <= mag[i-1] && mag[i] <= mag[i+1] && mag[i] < nullDepth*peak {
			nullIdx = i
			break
		}
	}
	if nullIdx < 0 {
		// No deep local minimum found; fall back to the first index below
		// the depth threshold.
		for i := peakIdx + 1; i < half; i++ {
			if mag[i] < nullDepth*peak {
				nullIdx = i
				break
			}
		}
	}
	if nullIdx < 0 {
		nullIdx = half - 1
	}
	binsPerIndex := float64(n) / float64(L)
	m.mainLobeWidthBins = 2 * float64(nullIdx-peakIdx) * binsPerIndex

	// Highest sidelobe: maximum magnitude beyond the first null.
	side := 0.0
	for i := nullIdx + 1; i < half; i++ {
		if mag[i] > side {
			side = mag[i]
		}
	}
	if side > 0 {
		m.sidelobeDB = 20 * math.Log10(side/peak)
	} else {
		m.sidelobeDB = math.Inf(-1)
	}

	// Scalloping loss: attenuation of the window transform exactly half a
	// bin off the peak, the worst-case bin alignment for a single tone.
	halfBinIdx := peakIdx + L/(2*n)
	if halfBinIdx < half && mag[halfBinIdx] > 0 {
		m.scallopingLossDB = 20 * math.Log10(peak/mag[halfBinIdx])
	}

	return m
}

// leakageRatio computes the fraction of spectral power of x (segmented and
// windowed exactly as Pwelch would) lying outside the main lobes of the
// detected spectral peaks. sidelobeDB is the window's highest sidelobe
// level; it sets the floor above which a local maximum is considered a
// genuine peak rather than leakage from a stronger one.
func leakageRatio(x []float64, nfft, pad, noverlap int, wf func(int) []float64, mainLobeWidthBins, sidelobeDB float64) float64 {
	if len(x) == 0 {
		return 0
	}
	if len(x) < nfft {
		x = dsputils.ZeroPadF(x, nfft)
	}

	segs := Segment(x, nfft, noverlap)
	if len(segs) == 0 {
		return 0
	}

	lp := pad/2 + 1
	power := make([]float64, lp)
	for _, s := range segs {
		s = dsputils.ZeroPadF(s, pad)
		window.Apply(s, wf)
		pgram := fft.FFTReal(s)
		for j := range power {
			p := cmplx.Abs(pgram[j])
			power[j] += p * p
		}
	}

	var total float64
	for _, p := range power {
		total += p
	}
	if total == 0 {
		return 0
	}

	// Main lobe half width in bins of the padded FFT. Rounding (rather
	// than ceiling) keeps the width stable against sub-bin measurement
	// noise in the null position.
	halfWidth := int(math.Round(mainLobeWidthBins / 2 * float64(pad) / float64(nfft)))
	if halfWidth < 1 {
		halfWidth = 1
	}

	// Greedily mark the main lobe region of each significant peak. The
	// region is centered on the sub-bin peak position (estimated by
	// quadratic interpolation) so that off-bin tones, whose main lobe
	// straddles two bins asymmetrically, are measured correctly.
	//
	// A local maximum only counts as a genuine peak if its power exceeds
	// what leakage from the strongest peak could produce at any single
	// bin (the window's sidelobe level, with a margin for coherent
	// sidelobe superposition). Below that floor a bump cannot be
	// distinguished from leakage, so it is counted as leakage.
	const maxPeaks = 16
	const sidelobeMargin = 4.0 // 6 dB above the window's sidelobe level
	const absoluteFloor = 1e-8 // -80 dB relative to the strongest peak
	inside := make([]bool, lp)
	strongest := 0.0
	for _, p := range power {
		if p > strongest {
			strongest = p
		}
	}
	peakFloor := strongest * math.Pow(10, sidelobeDB/10) * sidelobeMargin
	if floor := strongest * absoluteFloor; peakFloor < floor {
		peakFloor = floor
	}

	for peaks := 0; peaks < maxPeaks; peaks++ {
		best, bestVal := -1, 0.0
		for j, p := range power {
			if !inside[j] && p > bestVal {
				best, bestVal = j, p
			}
		}
		if best < 0 || bestVal < peakFloor {
			break
		}

		center := subBinCenter(power, best)
		lo := int(math.Ceil(center - float64(halfWidth)))
		hi := int(math.Floor(center + float64(halfWidth)))
		if lo < 0 {
			lo = 0
		}
		if hi >= lp {
			hi = lp - 1
		}
		for j := lo; j <= hi; j++ {
			inside[j] = true
		}
	}

	var confined float64
	for j, p := range power {
		if inside[j] {
			confined += p
		}
	}

	return 1 - confined/total
}

// subBinCenter estimates the sub-bin position of the peak at bin p of a
// power spectrum by quadratic interpolation. Returns p at the spectrum
// edges or when the interpolation is degenerate.
func subBinCenter(power []float64, p int) float64 {
	if p <= 0 || p >= len(power)-1 {
		return float64(p)
	}
	y0, y1, y2 := power[p-1], power[p], power[p+1]
	den := y0 - 2*y1 + y2
	if den >= 0 {
		// Degenerate or convex at a maximum; cannot interpolate.
		return float64(p)
	}
	delta := 0.5 * (y0 - y2) / den
	if delta < -1 {
		delta = -1
	} else if delta > 1 {
		delta = 1
	}
	return float64(p) + delta
}
