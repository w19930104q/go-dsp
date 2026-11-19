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
)

// Magnitude returns the magnitude (absolute value) of each bin of the
// spectrum x.
//
// Equivalent to numpy.abs applied to a complex spectrum.
func Magnitude(x []complex128) []float64 {
	r := make([]float64, len(x))
	for i, v := range x {
		r[i] = cmplx.Abs(v)
	}

	return r
}

// Phase returns the phase angle, in radians, of each bin of the spectrum x.
//
// Equivalent to numpy.angle applied to a complex spectrum.
func Phase(x []complex128) []float64 {
	r := make([]float64, len(x))
	for i, v := range x {
		r[i] = cmplx.Phase(v)
	}

	return r
}

// FromPolar returns the complex spectrum with the given per-bin magnitudes
// and phase angles (in radians). mag and phase must have equal length. It is
// the inverse of applying Magnitude and Phase, so spectra can be modified in
// the magnitude/phase domain and converted back for an inverse transform.
func FromPolar(mag, phase []float64) []complex128 {
	if len(mag) != len(phase) {
		panic("arrays not of equal size")
	}

	r := make([]complex128, len(mag))
	for i := range r {
		r[i] = cmplx.Rect(mag[i], phase[i])
	}

	return r
}
