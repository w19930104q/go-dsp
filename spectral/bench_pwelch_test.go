package spectral

import "testing"

func BenchmarkPwelchLong(b *testing.B) {
	x := make([]float64, 1<<16)
	for i := range x {
		x[i] = float64(i%97) * 0.5
	}
	o := &PwelchOptions{}
	b.ReportAllocs()
	for b.Loop() {
		Pwelch(x, 44100, o)
	}
}
