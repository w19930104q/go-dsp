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
	"runtime"
	"sync"

	"github.com/madelynnblue/go-dsp/dsputils"
)

// FFTBatch returns the forward FFT of each frame (row) of x. It is intended
// for multi-channel, multi-frame data: each row is one frame of one channel,
// and rows need not have the same length. The result for each row is
// bit-identical to calling FFT on that row directly, but a batch of frames
// is computed faster than calling FFT once per frame because frames are
// distributed across workers instead of paying the per-call parallelization
// overhead for every frame.
func FFTBatch(x [][]complex128) [][]complex128 {
	return batchApply(x, FFT, fftSequential)
}

// FFTRealBatch returns the forward FFT of each real-valued frame (row) of x.
// The result for each row is bit-identical to calling FFTReal on that row.
func FFTRealBatch(x [][]float64) [][]complex128 {
	return batchApply(dsputils.ToComplex2(x), FFT, fftSequential)
}

// IFFTBatch returns the inverse FFT of each frame (row) of x. The result for
// each row is bit-identical to calling IFFT on that row directly.
func IFFTBatch(x [][]complex128) [][]complex128 {
	return batchApply(x, IFFT, ifftSequential)
}

// IFFTRealBatch returns the inverse FFT of each real-valued frame (row) of x.
// The result for each row is bit-identical to calling IFFTReal on that row.
func IFFTRealBatch(x [][]float64) [][]complex128 {
	return batchApply(dsputils.ToComplex2(x), IFFT, ifftSequential)
}

// batchApply computes fn of every row of x. When there are at least as many
// rows as workers, rows are chunked across workers and each row is computed
// with seqFn (a single-threaded transform), avoiding per-frame goroutine and
// channel overhead. Otherwise there are few frames, so each row is computed
// with parFn (the regular, internally parallelized transform). Both paths
// produce bit-identical results for every row.
func batchApply(x [][]complex128, parFn, seqFn func([]complex128) []complex128) [][]complex128 {
	rows := len(x)
	r := make([][]complex128, rows)
	if rows == 0 {
		return r
	}

	numWorkers := worker_pool_size
	if numWorkers == 0 {
		numWorkers = runtime.GOMAXPROCS(0)
	}

	if rows < numWorkers {
		for i, row := range x {
			r[i] = parFn(row)
		}
		return r
	}

	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		start := w * rows / numWorkers
		end := (w + 1) * rows / numWorkers
		go func() {
			defer wg.Done()
			for i := start; i < end; i++ {
				r[i] = seqFn(x[i])
			}
		}()
	}
	wg.Wait()

	return r
}

// fftSequential returns the same result as FFT, computed single-threaded.
// The radix-2 butterflies and the Bluestein convolution perform exactly the
// same floating-point operations as their parallel counterparts, so the
// result is bit-identical to FFT(x).
func fftSequential(x []complex128) []complex128 {
	lx := len(x)

	if lx <= 1 {
		r := make([]complex128, lx)
		copy(r, x)
		return r
	}

	if dsputils.IsPowerOf2(lx) {
		return radix2FFTSequential(x)
	}

	return bluesteinFFTSequential(x)
}

// ifftSequential returns the same result as IFFT, computed single-threaded.
func ifftSequential(x []complex128) []complex128 {
	lx := len(x)
	r := make([]complex128, lx)

	// Reverse inputs, which is calculated with modulo N, hence x[0] as an outlier
	r[0] = x[0]
	for i := 1; i < lx; i++ {
		r[i] = x[lx-i]
	}

	r = fftSequential(r)

	N := complex(float64(lx), 0)
	for n := range r {
		r[n] /= N
	}
	return r
}

// radix2FFTSequential performs the same computations as radix2FFT on a
// single goroutine. Each butterfly is independent of how blocks are
// partitioned among workers, so the result is bit-identical to radix2FFT(x).
func radix2FFTSequential(x []complex128) []complex128 {
	lx := len(x)
	factors := getRadix2Factors(lx)

	t := make([]complex128, lx) // temp
	r := reorderData(x)

	for stage := 2; stage <= lx; stage <<= 1 {
		blocks := lx / stage
		s_2 := stage / 2

		if stage != 2 {
			for nb := 0; nb < lx; nb += stage {
				for j := 0; j < s_2; j++ {
					idx := j + nb
					idx2 := idx + s_2
					ridx := r[idx]
					w_n := r[idx2] * factors[blocks*j]
					t[idx] = ridx + w_n
					t[idx2] = ridx - w_n
				}
			}
		} else {
			for nb := 0; nb < lx; nb += 2 {
				n1 := nb + 1
				rn := r[nb]
				rn1 := r[n1]
				t[nb] = rn + rn1
				t[n1] = rn - rn1
			}
		}

		r, t = t, r
	}

	return r
}

// bluesteinFFTSequential performs the same computations as bluesteinFFT,
// using the single-threaded convolution below. The result is bit-identical
// to bluesteinFFT(x).
func bluesteinFFTSequential(x []complex128) []complex128 {
	lx := len(x)
	a := dsputils.ZeroPad(x, dsputils.NextPowerOf2(lx*2-1))
	la := len(a)
	factors, invFactors := getBluesteinFactors(lx)

	for n, v := range x {
		a[n] = v * invFactors[n]
	}

	b := make([]complex128, la)
	for i := 0; i < lx; i++ {
		b[i] = factors[i]

		if i != 0 {
			b[la-i] = factors[i]
		}
	}

	r := convolveSequential(a, b)

	for i := 0; i < lx; i++ {
		r[i] *= invFactors[i]
	}

	return r[:lx]
}

// convolveSequential performs the same computations as Convolve, using the
// single-threaded transforms. The result is bit-identical to Convolve(x, y).
func convolveSequential(x, y []complex128) []complex128 {
	if len(x) != len(y) {
		panic("arrays not of equal size")
	}

	fftX := fftSequential(x)
	fftY := fftSequential(y)

	r := make([]complex128, len(x))
	for i := 0; i < len(r); i++ {
		r[i] = fftX[i] * fftY[i]
	}

	return ifftSequential(r)
}
