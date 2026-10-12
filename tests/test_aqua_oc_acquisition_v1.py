"""Download integrity checks for the isolated Aqua-OC acquisition workflow."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_aqua_oc_data_v1.py"
    spec = importlib.util.spec_from_file_location("aqua_oc_acquisition_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CACHE", tmp_path / "cache")
    return module


def entry(payload):
    return {"key": "DOC.csv", "size": len(payload),
            "checksum": "md5:" + hashlib.md5(payload).hexdigest(),
            "links": {"self": "https://example.org/public/DOC.csv"}}


class Response:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload


def test_resume_validates_range_and_provider_checksum(acquisition, monkeypatch):
    payload = b"abc,def\n123,456\n"
    path = acquisition.CACHE / "raw/grqa/DOC.csv.partial"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload[:5])
    requests = []

    def request(url, **kwargs):
        requests.append(kwargs["headers"])
        return Response(payload[5:], 206, {"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}"})

    monkeypatch.setattr(acquisition.requests, "get", request)
    receipt = acquisition.download_file("grqa", entry(payload))
    assert requests[0]["Range"] == "bytes=5-"
    assert (acquisition.ROOT / receipt["file"]).read_bytes() == payload
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()


def test_ignored_range_replaces_partial_instead_of_appending(acquisition, monkeypatch):
    payload = b"complete file"
    path = acquisition.CACHE / "raw/grqa/DOC.csv.partial"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial")
    monkeypatch.setattr(acquisition.requests, "get", lambda *args, **kwargs: Response(payload))
    receipt = acquisition.download_file("grqa", entry(payload))
    assert (acquisition.ROOT / receipt["file"]).read_bytes() == payload


def test_corrupt_download_is_never_promoted_to_verified_file(acquisition, monkeypatch):
    monkeypatch.setattr(acquisition.requests, "get", lambda *args, **kwargs: Response(b"bad"))
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        acquisition.download_file("grqa", entry(b"yes"))
    assert not (acquisition.CACHE / "raw/grqa/DOC.csv").exists()


def test_incorrect_resume_offset_is_rejected(acquisition, monkeypatch):
    path = acquisition.CACHE / "raw/grqa/DOC.csv.partial"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc")
    monkeypatch.setattr(acquisition.requests, "get", lambda *args, **kwargs:
                        Response(b"def", 206, {"Content-Range": "bytes 2-4/6"}))
    with pytest.raises(RuntimeError, match="Invalid range response"):
        acquisition.download_file("grqa", entry(b"abcdef"))
    assert path.read_bytes() == b"abc"
