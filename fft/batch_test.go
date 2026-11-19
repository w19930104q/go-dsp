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
	"testing"
)

// makeFrames returns frames×size pseudo-random complex frames.
func makeFrames(frames, size int, rng *rand.Rand) [][]complex128 {
	x := make([][]complex128, frames)
	for i := range x {
		x[i] = make([]complex128, size)
		for j := range x[i] {
			x[i][j] = complex(rng.NormFloat64(), rng.NormFloat64())
		}
	}
	return x
}

// makeFramesReal returns frames×size pseudo-random real frames.
func makeFramesReal(frames, size int, rng *rand.Rand) [][]float64 {
	x := make([][]float64, frames)
	for i := range x {
		x[i] = make([]float64, size)
		for j := range x[i] {
			x[i][j] = rng.NormFloat64()
		}
	}
	return x
}

// requireBitIdentical fails the test unless got[i][j] == want[i][j] for all
// i, j, comparing bit-for-bit (no epsilon).
func requireBitIdentical(t *testing.T, got, want [][]complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("row count mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("row %d length mismatch: got %d, want %d", i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("row %d element %d not bit-identical: got %v, want %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

// TestFFTBatchBitIdentical verifies that for arbitrary frame counts and frame
// sizes (power-of-2 and not), the batch transforms are bit-identical to
// calling the per-frame transforms on every frame.
func TestFFTBatchBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	// frame counts: none, one, a few, and more than any plausible worker count
	frameCounts := []int{1, 2, 3, 7, 64, 257}
	// frame sizes: tiny, power-of-2, and non-power-of-2 (Bluestein path)
	sizes := []int{1, 2, 3, 4, 5, 8, 13, 16, 100, 256, 1000, 1024}

	for _, frames := range frameCounts {
		for _, size := range sizes {
			x := makeFrames(frames, size, rng)
			xr := makeFramesReal(frames, size, rng)

			wantFFT := make([][]complex128, frames)
			wantIFFT := make([][]complex128, frames)
			wantFFTReal := make([][]complex128, frames)
			wantIFFTReal := make([][]complex128, frames)
			for i := range x {
				wantFFT[i] = FFT(x[i])
				wantIFFT[i] = IFFT(x[i])
				wantFFTReal[i] = FFTReal(xr[i])
				wantIFFTReal[i] = IFFTReal(xr[i])
			}

			requireBitIdentical(t, FFTBatch(x), wantFFT)
			requireBitIdentical(t, IFFTBatch(x), wantIFFT)
			requireBitIdentical(t, FFTRealBatch(xr), wantFFTReal)
			requireBitIdentical(t, IFFTRealBatch(xr), wantIFFTReal)
		}
	}
}

// TestFFTBatchRagged verifies bit-identity when frames in one batch have
// different lengths (mixed radix-2 and Bluestein paths in a single call).
func TestFFTBatchRagged(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 16, 100, 256, 1000, 1024, 2048}

	x := make([][]complex128, len(sizes))
	for i, size := range sizes {
		x[i] = makeFrames(1, size, rng)[0]
	}

	got := FFTBatch(x)
	for i := range x {
		requireBitIdentical(t, got[i:i+1], [][]complex128{FFT(x[i])})
	}
}

// TestFFTBatchEmpty verifies that an empty batch yields an empty result.
func TestFFTBatchEmpty(t *testing.T) {
	if r := FFTBatch(nil); r == nil || len(r) != 0 {
		t.Error("FFTBatch(nil) should return an empty, non-nil slice")
	}
	if r := FFTRealBatch(nil); r == nil || len(r) != 0 {
		t.Error("FFTRealBatch(nil) should return an empty, non-nil slice")
	}
	if r := IFFTBatch(nil); r == nil || len(r) != 0 {
		t.Error("IFFTBatch(nil) should return an empty, non-nil slice")
	}
	if r := IFFTRealBatch(nil); r == nil || len(r) != 0 {
		t.Error("IFFTRealBatch(nil) should return an empty, non-nil slice")
	}
}

// TestFFTBatchDoesNotModifyInput verifies frames are not transformed in place.
func TestFFTBatchDoesNotModifyInput(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	x := makeFrames(16, 256, rng)
	backup := makeFrames(0, 0, rng)
	for _, row := range x {
		c := make([]complex128, len(row))
		copy(c, row)
		backup = append(backup, c)
	}

	FFTBatch(x)
	requireBitIdentical(t, x, backup)
}

const (
	benchFrames = 512 // e.g. 8 channels × 64 frames
	benchSize   = 1024
)

func benchFramesComplex() [][]complex128 {
	rng := rand.New(rand.NewSource(4))
	x := makeFrames(benchFrames, benchSize, rng)
	EnsureRadix2Factors(benchSize)
	return x
}

// BenchmarkFFTPerFrame is the baseline: one FFT call per frame per channel.
func BenchmarkFFTPerFrame(b *testing.B) {
	x := benchFramesComplex()
	b.ResetTimer()
	for b.Loop() {
		for _, frame := range x {
			FFT(frame)
		}
	}
}

// BenchmarkFFTBatch transforms the same multi-channel, multi-frame data in
// one call.
func BenchmarkFFTBatch(b *testing.B) {
	x := benchFramesComplex()
	b.ResetTimer()
	for b.Loop() {
		FFTBatch(x)
	}
}

const benchSizeBluestein = 1000

func benchFramesComplexBluestein() [][]complex128 {
	rng := rand.New(rand.NewSource(5))
	return makeFrames(benchFrames, benchSizeBluestein, rng)
}

// BenchmarkFFTPerFrameBluestein is the per-frame baseline for non-power-of-2
// frame sizes.
func BenchmarkFFTPerFrameBluestein(b *testing.B) {
	x := benchFramesComplexBluestein()
	b.ResetTimer()
	for b.Loop() {
		for _, frame := range x {
			FFT(frame)
		}
	}
}

// BenchmarkFFTBatchBluestein transforms non-power-of-2 frames in one call.
func BenchmarkFFTBatchBluestein(b *testing.B) {
	x := benchFramesComplexBluestein()
	b.ResetTimer()
	for b.Loop() {
		FFTBatch(x)
	}
}
