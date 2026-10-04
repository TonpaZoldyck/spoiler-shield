"""Reproducible raw-data downloads with checksums (tracker task P1.1).

Each source has a fixed URL and expected size. The SHA-256 of every file is
pinned in ``checksums.json`` the first time it is downloaded and verified on
every later download, so a silently changed upstream file fails loudly instead
of shifting results.

Raw files land in ``data/raw/`` at the repo root, which git ignores.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

CHECKSUMS = Path(__file__).with_name("checksums.json")
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    size: int | None
    """Expected size in bytes, when the host reports it."""
    citation: str
    extract: bool = False
    """Unzip the archive next to it after verification."""


GOODREADS_BASE = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"

SOURCES: dict[str, Source] = {
    "goodreads-spoilers": Source(
        name="goodreads-spoilers",
        url=GOODREADS_BASE + "goodreads_reviews_spoiler.json.gz",
        filename="goodreads_reviews_spoiler.json.gz",
        size=620_197_603,
        citation=(
            "Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley. "
            "Fine-Grained Spoiler Detection from Large-Scale Review Corpora. ACL 2019."
        ),
    ),
    "goodreads-works": Source(
        name="goodreads-works",
        url=GOODREADS_BASE + "goodreads_book_works.json.gz",
        filename="goodreads_book_works.json.gz",
        size=75_397_299,
        citation=(
            "Mengting Wan, Julian McAuley. "
            "Item Recommendation on Monotonic Behavior Chains. RecSys 2018."
        ),
    ),
    "imdb-spoilers": Source(
        name="imdb-spoilers",
        # Kaggle serves public datasets without an API key; this redirects to
        # Kaggle's storage bucket. Contains IMDB_reviews.json and
        # IMDB_movie_details.json.
        url="https://www.kaggle.com/api/v1/datasets/download/rmisra/imdb-spoiler-dataset",
        filename="imdb-spoiler-dataset.zip",
        size=347_575_137,
        citation="Rishabh Misra. IMDB Spoiler Dataset. arXiv:2212.06034, 2022.",
        extract=True,
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checksums() -> dict[str, str]:
    return json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}


def _save_checksums(sums: dict[str, str]) -> None:
    CHECKSUMS.write_text(json.dumps(dict(sorted(sums.items())), indent=2) + "\n")


class ChecksumMismatchError(RuntimeError):
    pass


def download(name: str, raw_dir: str | Path, *, force: bool = False) -> Path:
    """Download one source into ``raw_dir`` and verify it. Returns the file path."""
    src = SOURCES[name]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / src.filename
    sums = _load_checksums()

    if dest.exists() and not force and name in sums and sha256_of(dest) == sums[name]:
        print(f"{name}: already present and verified")
        _maybe_extract(src, dest)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"{name}: downloading {src.url}")
    with urllib.request.urlopen(src.url, timeout=60) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, _CHUNK)
    if src.size is not None and tmp.stat().st_size != src.size:
        tmp.unlink()
        raise ChecksumMismatchError(f"{name}: expected {src.size} bytes, got {tmp.stat().st_size}")

    digest = sha256_of(tmp)
    if name in sums and sums[name] != digest:
        tmp.unlink()
        raise ChecksumMismatchError(f"{name}: SHA-256 {digest} does not match pinned {sums[name]}")
    tmp.replace(dest)
    if name not in sums:
        sums[name] = digest
        _save_checksums(sums)
        print(f"{name}: pinned SHA-256 {digest} in {CHECKSUMS.name}")
    print(f"{name}: ok, {dest.stat().st_size:,} bytes")
    _maybe_extract(src, dest)
    return dest


def _maybe_extract(src: Source, archive: Path) -> None:
    if not src.extract:
        return
    with zipfile.ZipFile(archive) as zf:
        missing = [n for n in zf.namelist() if not (archive.parent / n).exists()]
        if missing:
            zf.extractall(archive.parent, members=missing)
            print(f"{src.name}: extracted {', '.join(missing)}")
