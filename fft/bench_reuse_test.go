package fft

import "testing"

func benchInput(n int) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(float64(i)*0.5, float64(i)*-0.25)
	}
	return x
}

func BenchmarkFFT4096Reuse(b *testing.B) {
	x := benchInput(4096)
	EnsureRadix2Factors(4096)
	b.ReportAllocs()
	for b.Loop() {
		FFT(x)
	}
}

func BenchmarkConvolve4096Reuse(b *testing.B) {
	x := benchInput(4096)
	y := benchInput(4096)
	EnsureRadix2Factors(4096)
	b.ReportAllocs()
	for b.Loop() {
		Convolve(x, y)
	}
}

func BenchmarkFFT1000Reuse(b *testing.B) {
	x := benchInput(1000)
	b.ReportAllocs()
	for b.Loop() {
		FFT(x)
	}
}
