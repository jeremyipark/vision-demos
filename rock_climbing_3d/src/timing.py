"""Timing instrumentation for the gateway call.

The gateway returns no server-timing header (only `x-request-id`), so every
number here is measured client-side. A single blocking call would only tell us
the round trip, which on a 1.7 MB upload is not the same thing as "how fast did
ViTPose run". This splits it:

    round trip = upload + server + download

`upload` is timed by wrapping the outbound byte stream and noting when its last
chunk is handed to the socket; `server` is the gap between that and the response
headers arriving. Both are approximations, since the kernel buffers and a small
body can look like it uploaded instantly, but on a multi-megabyte video the
split is meaningful. It is the difference between blaming the model and blaming
the network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def _httpx():
    """The httpx the installed openai actually uses (3.x vendors `httpx2`)."""
    try:
        import httpx2

        return httpx2
    except ImportError:
        import httpx

        return httpx


def _timed_stream_class(httpx):
    """Build the wrapper against the installed httpx's own stream base class.

    ``HTTPTransport.handle_request`` asserts ``isinstance(request.stream,
    SyncByteStream)``, so a duck-typed wrapper is rejected outright. The class
    has to be built once the module is resolved.
    """

    class _TimedStream(httpx.SyncByteStream):
        """Pass a request body through, recording when the last chunk is consumed."""

        def __init__(self, stream, on_done):
            self._stream = stream
            self._on_done = on_done

        def __iter__(self):
            for chunk in self._stream:
                yield chunk
            self._on_done()

        def close(self) -> None:
            close = getattr(self._stream, "close", None)
            if close:
                close()

    return _TimedStream


@dataclass
class RequestTiming:
    """What one HTTP round trip spent, in seconds."""

    round_trip: float = 0.0
    upload: float | None = None
    server: float | None = None
    download: float | None = None

    @property
    def measured_split(self) -> bool:
        return self.upload is not None and self.server is not None


def timed_http_client(timeout: float, sink: dict):
    """An httpx client that records upload / time-to-headers into *sink*.

    Returns None if the transport cannot be wrapped, in which case the caller
    falls back to round-trip timing alone rather than failing the run.
    """
    httpx = _httpx()

    try:
        timed_stream = _timed_stream_class(httpx)

        class _Transport(httpx.HTTPTransport):
            def handle_request(self, request):
                t0 = time.perf_counter()
                marks: dict[str, float] = {}

                stream = getattr(request, "stream", None)
                if stream is not None:
                    request.stream = timed_stream(
                        stream, lambda: marks.setdefault("uploaded", time.perf_counter())
                    )

                response = super().handle_request(request)
                headers_at = time.perf_counter()

                uploaded_at = marks.get("uploaded", headers_at)
                sink["upload"] = uploaded_at - t0
                sink["headers"] = headers_at - t0
                return response

        return httpx.Client(transport=_Transport(), timeout=timeout)
    except Exception:
        return None


def summarize(sink: dict, round_trip: float) -> RequestTiming:
    """Fold the transport's marks into a RequestTiming."""
    timing = RequestTiming(round_trip=round_trip)
    if "upload" in sink and "headers" in sink:
        timing.upload = sink["upload"]
        timing.server = max(0.0, sink["headers"] - sink["upload"])
        timing.download = max(0.0, round_trip - sink["headers"])
    return timing


@dataclass
class Metrics:
    """Per-phase wall clock and the throughput figures derived from it."""

    frames: int
    video_seconds: float
    convert_seconds: float
    convert_cached: bool
    encode_seconds: float
    request: RequestTiming = field(default_factory=RequestTiming)
    render_seconds: float = 0.0
    total_seconds: float = 0.0
    upload_mb: float = 0.0
    cost_usd: float | None = None
    poses_cached: bool = False

    @property
    def pose_seconds(self) -> float:
        """The server's share, the closest thing we have to ViTPose's own time."""
        return self.request.server if self.request.server is not None else self.request.round_trip

    def _fps(self, seconds: float) -> float:
        return self.frames / seconds if seconds > 0 else 0.0

    def as_dict(self) -> dict:
        pose_s = self.pose_seconds
        out = {
            "frames": self.frames,
            "video_seconds": round(self.video_seconds, 3),
            "pose": {
                "seconds": round(pose_s, 3),
                "fps": round(self._fps(pose_s), 2),
                "ms_per_frame": round(pose_s / self.frames * 1000, 2) if self.frames else 0.0,
                "realtime_factor": round(self.video_seconds / pose_s, 2) if pose_s else 0.0,
                "measured": "reused from cache; timings are from the original run"
                if self.poses_cached
                else "server-side share of the round trip"
                if self.request.measured_split
                else "round trip (upload not separable on this transport)",
                "from_cache": self.poses_cached,
            },
            "gateway_round_trip": {
                "seconds": round(self.request.round_trip, 3),
                "fps": round(self._fps(self.request.round_trip), 2),
                "upload_seconds": round(self.request.upload, 3)
                if self.request.upload is not None else None,
                "server_seconds": round(self.request.server, 3)
                if self.request.server is not None else None,
                "download_seconds": round(self.request.download, 3)
                if self.request.download is not None else None,
                "upload_mb": round(self.upload_mb, 2),
            },
            "local": {
                "convert_seconds": round(self.convert_seconds, 3),
                "convert_cached": self.convert_cached,
                "render_seconds": round(self.render_seconds, 3),
                "render_fps": round(self._fps(self.render_seconds), 2),
                "encode_seconds": round(self.encode_seconds, 3),
            },
            "total_seconds": round(self.total_seconds, 3),
            "total_fps": round(self._fps(self.total_seconds), 2),
        }
        if self.cost_usd is not None:
            out["cost"] = {
                "usd": self.cost_usd,
                "usd_per_frame": round(self.cost_usd / self.frames, 8) if self.frames else 0.0,
                "usd_per_video_second": round(self.cost_usd / self.video_seconds, 6)
                if self.video_seconds else 0.0,
                "usd_per_1000_frames": round(self.cost_usd / self.frames * 1000, 4)
                if self.frames else 0.0,
            }
        return out

    def as_text(self) -> str:
        d = self.as_dict()
        pose, trip, local = d["pose"], d["gateway_round_trip"], d["local"]

        lines = [
            "ViTPose video performance",
            "=" * 52,
            f"{'frames':<26}{d['frames']}",
            f"{'video duration':<26}{d['video_seconds']:.2f}s",
            "",
            "ViTPose (server-side)" if not self.poses_cached
            else "ViTPose (server-side), REUSED FROM CACHE",
            "-" * 52,
            f"{'  seconds':<26}{pose['seconds']:.2f}s",
            f"{'  throughput':<26}{pose['fps']:.2f} fps",
            f"{'  per frame':<26}{pose['ms_per_frame']:.1f} ms",
            f"{'  vs. realtime':<26}{pose['realtime_factor']:.2f}x",
            f"{'  measured as':<26}{pose['measured']}",
            "",
            "Gateway round trip",
            "-" * 52,
            f"{'  total':<26}{trip['seconds']:.2f}s  ({trip['fps']:.2f} fps end-to-end)",
        ]
        if trip["upload_seconds"] is not None:
            lines += [
                f"{'  upload':<26}{trip['upload_seconds']:.2f}s  ({trip['upload_mb']:.1f} MB)",
                f"{'  server':<26}{trip['server_seconds']:.2f}s",
                f"{'  download + parse':<26}{trip['download_seconds']:.2f}s",
            ]
        lines += [
            "",
            "Local",
            "-" * 52,
            f"{'  convert':<26}{local['convert_seconds']:.2f}s"
            + ("  (cache hit)" if local["convert_cached"] else ""),
            f"{'  render':<26}{local['render_seconds']:.2f}s  ({local['render_fps']:.2f} fps)",
            f"{'  encode':<26}{local['encode_seconds']:.2f}s",
            "",
            f"{'total':<26}{d['total_seconds']:.2f}s  ({d['total_fps']:.2f} fps)",
        ]
        if "cost" in d:
            c = d["cost"]
            lines += [
                "",
                "Cost",
                "-" * 52,
                f"{'  this run':<26}${c['usd']:.6f}",
                f"{'  per 1000 frames':<26}${c['usd_per_1000_frames']:.4f}",
                f"{'  per video second':<26}${c['usd_per_video_second']:.6f}",
            ]
        return "\n".join(lines) + "\n"
