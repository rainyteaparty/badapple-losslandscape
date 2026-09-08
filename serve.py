"""Static server with HTTP Range support.

`python -m http.server` does not implement Range. Chrome then reports the
heightmap clip as non-seekable (`seekable.end(0) == 0`) even once it is fully
buffered, so the viewer's scrub bar does nothing and the source/atlas videos,
which are kept in step by assigning `currentTime`, stall at zero. Serving with
Range makes all of that work.
"""
import argparse, os, re, sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        m = RANGE.match(rng.strip())
        if not m:
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404); return None
        size = os.fstat(f.fileno()).st_size
        lo, hi = m.group(1), m.group(2)
        if lo == "":                                  # suffix form: bytes=-N
            length = min(int(hi or 0), size); start = size - length
        else:
            start = int(lo)
            end = int(hi) if hi else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers(); f.close(); return None
            length = end - start + 1
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{start+length-1}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return _Limited(f, length)

    def end_headers(self):
        if not self.headers.get("Range"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


class _Limited:
    """File wrapper that stops after `remaining` bytes, for copyfile()."""
    def __init__(self, f, remaining): self.f, self.remaining = f, remaining
    def read(self, n=-1):
        if self.remaining <= 0: return b""
        if n < 0 or n > self.remaining: n = self.remaining
        b = self.f.read(n); self.remaining -= len(b); return b
    def close(self): self.f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    ThreadingHTTPServer(("", a.port), partial(RangeHandler,
        directory=os.path.dirname(os.path.abspath(__file__)))).serve_forever()
