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
	"testing"

	"github.com/madelynnblue/go-dsp/dsputils"
	"github.com/madelynnblue/go-dsp/fft"
)

// Reference values generated with NumPy 2.5.4 from the fixed sine + noise
// signal below (see fft/rfft_test.go for the generator):
//
//	spectrum = np.fft.rfft(x)
//	magnitude = np.abs(spectrum)
//	phase = np.angle(spectrum)
var polarTests = []struct {
	in        []float64
	magnitude []float64
	phase     []float64
}{
	// N = 16 (even, power of 2)
	{
		in:        []float64{0.356600561390818, 1.1048214076314269, 0.21461129314461214, -0.18193606901863296, -0.6659569612932642, -0.8576449936893568, 0.70838518521822, 1.3826568728434945, -0.3537214021683163, -1.1237516879695668, -0.19831280143792018, 0.1991196355368337, 0.6471069163823377, 0.8558956106904141, -0.7024316877640266, -1.3944122233957628},
		magnitude: []float64{0.008970343898689559, 0.043777796059895895, 0.021600738056232464, 8.028860131591692, 0.04628833031603996, 4.016738966024762, 0.054378772157174605, 0.004194752606035334, 0.02153255084361072},
		phase:     []float64{3.141592653589793, 1.3897895296168439, -2.0987873475203127, -1.5739122342213019, 2.5423378949843154, -0.7795253674614255, 0.03735448686936004, -1.1975573203914278, 0.0},
	},
	// N = 15 (odd, non power of 2)
	{
		in:        []float64{0.356600561390818, 1.0700661977840091, 0.11232685110600361, -0.22482621453528753, -0.8411573456304308, -0.4959847082131585, 1.3058883109200992, 0.7140323489202989, -1.0709161770120499, -0.6060335649776157, 0.1382035022998858, 0.4758715225049091, 0.9419989498613581, -0.44710331767153066, -1.4293443360171676},
		magnitude: []float64{0.000377419269859125, 0.04895186112271337, 0.007130389183114963, 7.511886920394374, 0.05845802137604561, 3.744286544603053, 0.03846340113918024, 0.012451575698530351},
		phase:     []float64{3.141592653589793, 1.0118923759203255, -1.941514713344592, -1.5747176165951862, 1.8582973350718528, -0.7804682105831262, -0.8037768384299643, 0.6345998327315517},
	},
}

func TestMagnitudePhase(t *testing.T) {
	for _, pt := range polarTests {
		spectrum := fft.RFFT(pt.in)

		if v := Magnitude(spectrum); !dsputils.PrettyClose(v, pt.magnitude) {
			t.Error("Magnitude error\ninput:", spectrum, "\noutput:", v, "\nexpected:", pt.magnitude)
		}

		if v := Phase(spectrum); !dsputils.PrettyClose(v, pt.phase) {
			t.Error("Phase error\ninput:", spectrum, "\noutput:", v, "\nexpected:", pt.phase)
		}
	}
}

// TestPolarChain verifies the full spectral processing chain: RFFT, split
// into magnitude and phase, recombine with FromPolar, and invert with IRFFT
// to exactly recover the input signal.
func TestPolarChain(t *testing.T) {
	for _, pt := range polarTests {
		spectrum := fft.RFFT(pt.in)

		recombined := FromPolar(Magnitude(spectrum), Phase(spectrum))
		if !dsputils.PrettyCloseC(recombined, spectrum) {
			t.Error("FromPolar error\ninput:", spectrum, "\noutput:", recombined)
		}

		v := fft.IRFFT(recombined, len(pt.in))
		if !dsputils.PrettyClose(v, pt.in) {
			t.Error("polar chain round trip error\ninput:", pt.in, "\noutput:", v)
		}
	}
}
