"""src/cache_utils.py — tiny shared helpers for the draft-head cache format.

The draft-head scripts live in a hyphenated folder (`draft-head/`), which is
not a valid Python import path, so anything they share lives here in `src/`
(the importable package). This is the chunk-index parser for the feature-cache
filenames written by draft-head/collect_features.py.
"""

import re
from pathlib import Path


def chunk_idx(path: Path) -> int:
    """'chunk_000042.features.pt' -> 42.

    Regex on the full filename (NOT Path.stem — for `chunk_000000.features.pt`
    the stem is `chunk_000000.features`, which broke naive `split("_")`).
    """
    m = re.search(r"chunk_(\d+)", path.name)
    if m is None:
        raise ValueError(f"unexpected chunk filename: {path.name}")
    return int(m.group(1))
