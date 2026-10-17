from pipeline.gemini import GeminiRequest, Image
from pipeline.gemini.sdk import _chunks_by_size


def _req(n_bytes: int) -> GeminiRequest:
    return GeminiRequest(prompt="", images=(Image(b"x" * n_bytes, "image/webp"),))


def test_chunks_respect_size_limit_and_keep_order() -> None:
    requests = [_req(4), _req(4), _req(4), _req(9), _req(1)]

    chunks = [list(c) for c in _chunks_by_size(requests, max_bytes=10)]

    assert [len(c) for c in chunks] == [2, 1, 2]
    assert [r for c in chunks for r in c] == requests


def test_oversized_request_gets_its_own_chunk() -> None:
    chunks = list(_chunks_by_size([_req(50), _req(1)], max_bytes=10))

    assert [len(c) for c in chunks] == [1, 1]
