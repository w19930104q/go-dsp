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
	"testing"

	"github.com/madelynnblue/go-dsp/dsputils"
)

// Reference values generated with NumPy 2.5.4:
//
//	rng = np.random.default_rng(42)
//	noise = 0.01 * rng.standard_normal(N)
//	x = sin(2π·3n/N) + 0.5·sin(2π·5n/N + π/4) + noise
//	spectrum = np.fft.rfft(x)
//	irfft = np.fft.irfft(spectrum, n=N)
var rfftTests = []struct {
	in       []float64
	spectrum []complex128
	irfft    []float64
}{
	// N = 16 (even, power of 2)
	{
		in: []float64{0.356600561390818, 1.1048214076314269, 0.21461129314461214, -0.18193606901863296, -0.6659569612932642, -0.8576449936893568, 0.70838518521822, 1.3826568728434945, -0.3537214021683163, -1.1237516879695668, -0.19831280143792018, 0.1991196355368337, 0.6471069163823377, 0.8558956106904141, -0.7024316877640266, -1.3944122233957628},
		spectrum: []complex128{
			complex(-0.008970343898689559, 0.0),
			complex(0.007880879441039323, 0.04306259591684691),
			complex(-0.010882429382711917, -0.018659169737806073),
			complex(-0.025017144428145066, -8.028821156007101),
			complex(-0.038222874849310084, 0.026107879303015058),
			complex(2.8568945725534594, -2.8235341900703554),
			complex(0.054340837649568235, 0.0020308187671910748),
			complex(0.001529546670183568, -0.003905948848816454),
			complex(0.02153255084361072, 0.0)},
		irfft: []float64{0.35660056139081797, 1.1048214076314267, 0.21461129314461214, -0.1819360690186329, -0.6659569612932643, -0.8576449936893564, 0.7083851852182201, 1.3826568728434947, -0.35372140216831627, -1.1237516879695666, -0.19831280143792018, 0.19911963553683365, 0.647106916382338, 0.8558956106904139, -0.7024316877640266, -1.3944122233957628},
	},
	// N = 15 (odd, non power of 2)
	{
		in: []float64{0.356600561390818, 1.0700661977840091, 0.11232685110600361, -0.22482621453528753, -0.8411573456304308, -0.4959847082131585, 1.3058883109200992, 0.7140323489202989, -1.0709161770120499, -0.6060335649776157, 0.1382035022998858, 0.4758715225049091, 0.9419989498613581, -0.44710331767153066, -1.4293443360171676},
		spectrum: []complex128{
			complex(-0.000377419269859125, 0.0),
			complex(0.025957079056699237, 0.041503189687320204),
			complex(-0.0025832338296991053, -0.00664600277488514),
			complex(-0.02945621007276664, -7.511829167152314),
			complex(-0.016576163115729643, 0.05605863965137693),
			complex(2.6606307736239874, -2.6345256906207033),
			complex(0.026693308401559344, -0.027692968670273196),
			complex(0.010027366002013902, 0.007381982690164435)},
		irfft: []float64{0.35660056139081797, 1.070066197784009, 0.11232685110600359, -0.2248262145352875, -0.8411573456304305, -0.4959847082131585, 1.3058883109200992, 0.7140323489202987, -1.0709161770120494, -0.6060335649776156, 0.13820350229988604, 0.4758715225049088, 0.9419989498613579, -0.4471033176715309, -1.4293443360171674},
	},
}

func TestRFFT(t *testing.T) {
	for _, rt := range rfftTests {
		v := RFFT(rt.in)

		if len(v) != len(rt.in)/2+1 {
			t.Fatal("RFFT error: wrong output length\ninput:", len(rt.in), "\noutput:", len(v), "\nexpected:", len(rt.in)/2+1)
		}

		if !dsputils.PrettyCloseC(v, rt.spectrum) {
			t.Error("RFFT error\ninput:", rt.in, "\noutput:", v, "\nexpected:", rt.spectrum)
		}
	}
}

func TestIRFFT(t *testing.T) {
	for _, rt := range rfftTests {
		v := IRFFT(rt.spectrum, len(rt.in))

		if !dsputils.PrettyClose(v, rt.irfft) {
			t.Error("IRFFT error\ninput:", rt.spectrum, "\noutput:", v, "\nexpected:", rt.irfft)
		}
	}
}

// TestRFFTRoundTrip verifies that IRFFT(RFFT(x)) exactly recovers x, with
// both the explicit and default output lengths.
func TestRFFTRoundTrip(t *testing.T) {
	for _, rt := range rfftTests {
		v := IRFFT(RFFT(rt.in), len(rt.in))
		if !dsputils.PrettyClose(v, rt.in) {
			t.Error("RFFT round trip error\ninput:", rt.in, "\noutput:", v)
		}

		if len(rt.in)%2 == 0 {
			v = IRFFT(RFFT(rt.in), 0)
			if !dsputils.PrettyClose(v, rt.in) {
				t.Error("RFFT round trip (default n) error\ninput:", rt.in, "\noutput:", v)
			}
		}
	}
}
