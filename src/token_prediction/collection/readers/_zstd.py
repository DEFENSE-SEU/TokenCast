from __future__ import annotations


class ZstdFramesError(ValueError):
    pass


class ZstdUnavailableError(RuntimeError):
    pass


def decompress_zstd_frames(data: bytes) -> bytes:
    try:
        import zstandard
    except ImportError as exc:
        raise ZstdUnavailableError("zstandard is unavailable") from exc

    remaining = data
    decoded: list[bytes] = []
    frame_ordinal = 0
    while remaining:
        frame_ordinal += 1
        decompressor = zstandard.ZstdDecompressor().decompressobj()
        try:
            decoded.append(decompressor.decompress(remaining))
        except zstandard.ZstdError as exc:
            raise ZstdFramesError(
                f"Zstandard frame {frame_ordinal} is invalid: {exc}"
            ) from exc
        if not decompressor.eof:
            raise ZstdFramesError(f"Zstandard frame {frame_ordinal} is incomplete")
        unused = decompressor.unused_data
        if len(unused) >= len(remaining):
            raise ZstdFramesError(f"Zstandard frame {frame_ordinal} made no progress")
        remaining = unused
    if frame_ordinal == 0:
        raise ZstdFramesError("compressed session log is empty")
    return b"".join(decoded)
