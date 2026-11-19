/*
 * Copyright (c) 2011 Madelynn Blue <blue.mlynn@gmail.com>
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

package fft

import (
	"math/cmplx"

	"github.com/madelynnblue/go-dsp/dsputils"
)

// RFFT returns the forward FFT of the real-valued slice x, computed as the
// non-redundant half of the spectrum. The result has length len(x)/2+1 and
// contains the frequency bins 0..len(x)/2 (the Nyquist bin included for even
// lengths). The omitted bins are the complex conjugates of the returned ones
// and carry no additional information for real-valued input.
//
// Equivalent to numpy.fft.rfft.
func RFFT(x []float64) []complex128 {
	lx := len(x)
	if lx == 0 {
		return nil
	}

	return FFT(dsputils.ToComplex(x))[:lx/2+1]
}

// IRFFT returns the inverse FFT of the non-redundant spectrum x, as produced
// by RFFT. The missing bins are reconstructed by Hermitian symmetry and the
// real part of the full inverse transform is returned, yielding n real
// samples. If n <= 0, n defaults to 2*(len(x)-1), the even length implied by
// len(x). n must satisfy len(x) == n/2+1.
//
// Equivalent to numpy.fft.irfft.
func IRFFT(x []complex128, n int) []float64 {
	lx := len(x)
	if lx == 0 {
		return nil
	}

	if n <= 0 {
		n = 2 * (lx - 1)
	}

	if lx != n/2+1 {
		panic("IRFFT: len(x) != n/2+1")
	}

	full := make([]complex128, n)
	copy(full, x)
	for k := lx; k < n; k++ {
		full[k] = cmplx.Conj(full[n-k])
	}

	t := IFFT(full)
	r := make([]float64, n)
	for i, v := range t {
		r[i] = real(v)
	}

	return r
}
