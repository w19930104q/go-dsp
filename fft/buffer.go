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

import "sync"

// bufPool recycles scratch buffers used during FFT computation. Repeated
// transforms of the same size reuse buffers instead of allocating new ones,
// keeping the allocation count bounded and independent of the input length.
// sync.Pool is safe for concurrent use by multiple goroutines.
var bufPool = sync.Pool{
	New: func() any { return []complex128(nil) },
}

// getBuf returns a scratch buffer of length n. Its contents are unspecified;
// callers must fully overwrite any portion they rely on. A buffer obtained
// from getBuf must never be returned to callers of this package; release it
// with putBuf when done.
func getBuf(n int) []complex128 {
	if b, ok := bufPool.Get().([]complex128); ok && cap(b) >= n {
		return b[:n]
	}

	return make([]complex128, n)
}

// putBuf returns a scratch buffer to the pool for reuse. The buffer must not
// be referenced by the caller (or anyone else) after this call.
func putBuf(b []complex128) {
	if cap(b) == 0 {
		return
	}

	bufPool.Put(b[:cap(b)])
}
