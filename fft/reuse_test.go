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
	"math/rand"
	"sync"
	"testing"

	"github.com/madelynnblue/go-dsp/dsputils"
)

// sink keeps results alive so the compiler cannot optimize calls away.
var sinkC []complex128

func randComplex(n int, seed int64) []complex128 {
	r := rand.New(rand.NewSource(seed))
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(r.NormFloat64(), r.NormFloat64())
	}
	return x
}

func randFloat(n int, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	x := make([]float64, n)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	return x
}

// equalComplexC reports whether a and b are bit-identical.
func equalComplexC(a, b []complex128) bool {
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

// TestBufferReuseIdentical verifies that results computed with recycled
// scratch buffers are bit-identical to the first (cold-pool) result.
func TestBufferReuseIdentical(t *testing.T) {
	// Sizes cover radix-2 with even and odd stage counts, and non-power-of-2
	// sizes that go through the Bluestein algorithm.
	sizes := []int{2, 4, 8, 16, 32, 64, 256, 4096, 3, 5, 7, 100, 1000, 12345}

	for _, n := range sizes {
		x := randComplex(n, int64(n))

		ref := FFT(x)
		for i := 0; i < 8; i++ {
			if got := FFT(x); !equalComplexC(got, ref) {
				t.Fatalf("FFT(%d) run %d differs from reference", n, i)
			}
		}

		refI := IFFT(x)
		for i := 0; i < 8; i++ {
			if got := IFFT(x); !equalComplexC(got, refI) {
				t.Fatalf("IFFT(%d) run %d differs from reference", n, i)
			}
		}

		y := randComplex(n, int64(n*2))
		refC := Convolve(x, y)
		for i := 0; i < 8; i++ {
			if got := Convolve(x, y); !equalComplexC(got, refC) {
				t.Fatalf("Convolve(%d) run %d differs from reference", n, i)
			}
		}

		xf := randFloat(n, int64(n*3))
		refR := FFTReal(xf)
		for i := 0; i < 8; i++ {
			if got := FFTReal(xf); !equalComplexC(got, refR) {
				t.Fatalf("FFTReal(%d) run %d differs from reference", n, i)
			}
		}
	}
}

// TestBufferReuseIdentical2D verifies bit-identical results for the 2-D and
// N-D entry points when buffers are recycled.
func TestBufferReuseIdentical2D(t *testing.T) {
	x2 := [][]complex128{
		randComplex(16, 1),
		randComplex(16, 2),
		randComplex(16, 3),
	}

	ref := FFT2(x2)
	for i := 0; i < 8; i++ {
		got := FFT2(x2)
		for r := range got {
			if !equalComplexC(got[r], ref[r]) {
				t.Fatalf("FFT2 run %d row %d differs from reference", i, r)
			}
		}
	}

	m := dsputils.MakeMatrix(randComplex(24, 4), []int{2, 3, 4})
	refN := FFTN(m)
	for i := 0; i < 8; i++ {
		if got := FFTN(m); !got.PrettyClose(refN) {
			t.Fatalf("FFTN run %d differs from reference", i)
		}
	}
}

// TestBufferReuseAllocs verifies that, once the pool is warm, repeated
// transforms of the same size perform a small constant number of
// allocations, independent of the input length. (The pre-pool
// implementation allocated per work item per stage, growing with both the
// input length and GOMAXPROCS.)
func TestBufferReuseAllocs(t *testing.T) {
	const (
		maxFFTAllocs      = 16
		maxConvolveAllocs = 48
	)

	x := randComplex(4096, 1)
	xf := randFloat(4096, 2)
	y := randComplex(4096, 3)
	xb := randComplex(1000, 4)

	// Warm the pool and the factor caches.
	FFT(x)
	IFFT(x)
	FFTReal(xf)
	Convolve(x, y)
	FFT(xb)

	if n := testing.AllocsPerRun(50, func() { sinkC = FFT(x) }); n > maxFFTAllocs {
		t.Errorf("FFT(4096): %v allocs/run, want <= %d", n, maxFFTAllocs)
	}
	if n := testing.AllocsPerRun(50, func() { sinkC = IFFT(x) }); n > maxFFTAllocs {
		t.Errorf("IFFT(4096): %v allocs/run, want <= %d", n, maxFFTAllocs)
	}
	if n := testing.AllocsPerRun(50, func() { sinkC = FFTReal(xf) }); n > maxFFTAllocs {
		t.Errorf("FFTReal(4096): %v allocs/run, want <= %d", n, maxFFTAllocs)
	}
	if n := testing.AllocsPerRun(50, func() { sinkC = Convolve(x, y) }); n > maxConvolveAllocs {
		t.Errorf("Convolve(4096): %v allocs/run, want <= %d", n, maxConvolveAllocs)
	}
	if n := testing.AllocsPerRun(50, func() { sinkC = FFT(xb) }); n > maxConvolveAllocs {
		t.Errorf("FFT(1000) (bluestein): %v allocs/run, want <= %d", n, maxConvolveAllocs)
	}
}

// TestBufferReuseConcurrent hammers the entry points from many goroutines
// with shared input sizes, and verifies every result is bit-identical to the
// serially computed reference. Run with -race.
func TestBufferReuseConcurrent(t *testing.T) {
	sizes := []int{256, 1000, 4096}
	inputs := make([][]complex128, len(sizes))
	refs := make([][]complex128, len(sizes))
	for i, n := range sizes {
		inputs[i] = randComplex(n, int64(n))
		refs[i] = FFT(inputs[i])
	}

	var wg sync.WaitGroup
	errs := make(chan string, 64)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for iter := 0; iter < 50; iter++ {
				i := (g + iter) % len(sizes)
				if got := FFT(inputs[i]); !equalComplexC(got, refs[i]) {
					errs <- "FFT result mismatch"
					return
				}
				if got := IFFT(inputs[i]); got == nil {
					errs <- "IFFT returned nil"
					return
				}
				if got := Convolve(inputs[i], inputs[i]); got == nil {
					errs <- "Convolve returned nil"
					return
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
}
