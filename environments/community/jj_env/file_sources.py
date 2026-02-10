"""
File Sources — Pluggable content providers for task generation
==============================================================

Abstracts where base repo files come from:
  - SyntheticFileSource: wraps the existing TEMPLATES (zero dependencies)
  - HFCodeFileSource: streams real Python code from HuggingFace datasets

The scramble ops already operate on arbitrary Dict[str, str], so any
FileSource can be dropped in transparently.
"""

from __future__ import annotations

import hashlib
import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FileSource(ABC):
    """Abstract base for providing base files to the scramble pipeline."""

    @abstractmethod
    def sample(self, seed: int, n_files: int = 1) -> Dict[str, str]:
        """Return {filename: content} for use as base_files."""


class SyntheticFileSource(FileSource):
    """Wraps the existing TEMPLATES registry — zero external dependencies."""

    def __init__(self):
        from .code_templates import TEMPLATES

        self._templates = TEMPLATES

    def sample(self, seed: int, n_files: int = 1) -> Dict[str, str]:
        rng = random.Random(seed)
        keys = list(self._templates.keys())
        # Prefer Python templates for richer scramble ops
        python_keys = [k for k in keys if "python" in k]
        pool = python_keys if python_keys else keys

        files: Dict[str, str] = {}
        for i in range(n_files):
            key = rng.choice(pool)
            generated = self._templates[key](seed + i)
            files.update(generated)
            # After first file, allow any template for variety
            pool = keys
        return files


class HFCodeFileSource(FileSource):
    """
    Streams real Python source files from a HuggingFace code dataset,
    caches a filtered pool in memory, and samples deterministically.

    Default dataset: bigcode/starcoderdata (Python subset via data_dir).
    Content column is 'content', path from 'max_stars_repo_path'.
    """

    # Map dataset → (content_column, path_column, load_dataset extra kwargs)
    _DATASET_PROFILES = {
        "bigcode/starcoderdata": {
            "content_col": "content",
            "path_col": "max_stars_repo_path",
            "load_kwargs": {"data_dir": "python"},
        },
        "codeparrot/github-code": {
            "content_col": "code",
            "path_col": "path",
            "load_kwargs": {"languages": ["Python"]},
        },
    }

    def __init__(
        self,
        dataset_name: str = "bigcode/starcoderdata",
        pool_size: int = 5000,
        min_lines: int = 30,
        max_lines: int = 300,
        min_functions: int = 2,
        max_line_length: int = 200,
        buffer_size: int = 10_000,
    ):
        self.dataset_name = dataset_name
        self.pool_size = pool_size
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_functions = min_functions
        self.max_line_length = max_line_length
        self.buffer_size = buffer_size
        self._pool: List[Dict[str, str]] = []

        profile = self._DATASET_PROFILES.get(dataset_name, {})
        self._content_col = profile.get("content_col", "content")
        self._path_col = profile.get("path_col", "path")
        self._load_kwargs = profile.get("load_kwargs", {})

    @property
    def is_loaded(self) -> bool:
        return len(self._pool) > 0

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def _passes_quality_filter(self, code: str) -> bool:
        """Check all quality criteria for a candidate file."""
        lines = code.split("\n")

        # Line count bounds
        if not (self.min_lines <= len(lines) <= self.max_lines):
            return False

        # No extremely long lines (skip minified code)
        if max(len(l) for l in lines) > self.max_line_length:
            return False

        # Must have enough top-level function definitions (matches find_functions regex)
        if sum(1 for l in lines if l.startswith("def ")) < self.min_functions:
            return False

        # Must compile (no syntax errors)
        try:
            compile(code, "<filter>", "exec")
        except SyntaxError:
            return False

        return True

    def load(self) -> None:
        """
        Stream dataset, apply quality filters, cache pool in memory.
        Call this once during setup (e.g. from JJEnv.setup()).
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HFCodeFileSource requires the `datasets` package. "
                "Install with: pip install datasets"
            )

        logger.info(
            "Loading HF code dataset '%s' (target pool_size=%d)...",
            self.dataset_name,
            self.pool_size,
        )

        ds = load_dataset(
            self.dataset_name,
            streaming=True,
            split="train",
            **self._load_kwargs,
        )
        ds = ds.shuffle(seed=0, buffer_size=self.buffer_size)

        seen_hashes: set = set()
        accepted = 0
        scanned = 0

        for example in ds:
            scanned += 1
            code = example.get(self._content_col, "")

            # UTF-8 clean check (streaming usually handles this, but be safe)
            if not isinstance(code, str):
                continue

            # Dedup by content hash (stable across runs unlike builtin hash())
            h = hashlib.md5(code.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            if not self._passes_quality_filter(code):
                continue

            # Build a filename from the dataset path or fall back to index
            path = example.get(self._path_col, f"file_{accepted}.py")
            # Use just the basename to keep paths clean
            basename = path.rsplit("/", 1)[-1] if "/" in path else path
            if not basename.endswith(".py"):
                basename = f"file_{accepted}.py"

            self._pool.append({basename: code})
            accepted += 1

            if accepted % 500 == 0:
                logger.info(
                    "  HF pool progress: %d / %d accepted (scanned %d)",
                    accepted,
                    self.pool_size,
                    scanned,
                )

            if accepted >= self.pool_size:
                break

        logger.info(
            "HF code pool loaded: %d files accepted from %d scanned",
            len(self._pool),
            scanned,
        )

        if len(self._pool) == 0:
            raise RuntimeError(
                f"No files passed quality filters from '{self.dataset_name}'. "
                "Check dataset availability and filter settings."
            )

    def sample(self, seed: int, n_files: int = 1) -> Dict[str, str]:
        if not self._pool:
            raise RuntimeError(
                "HFCodeFileSource pool is empty. Call load() first."
            )

        rng = random.Random(seed)
        files: Dict[str, str] = {}

        for _ in range(n_files):
            idx = rng.randint(0, len(self._pool) - 1)
            entry = self._pool[idx]
            files.update(entry)

        return files
