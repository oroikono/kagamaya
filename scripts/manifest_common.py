"""Shared, dependency-free policy for a self-contained release inventory.

This is a local release check, not a defense against concurrent hostile changes
to the filesystem. Git metadata and the root-relative recount output are private
generated state. Cache names alone never exclude arbitrary source files.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess


class ManifestError(ValueError):
    """The bundle or its manifest is not safe to inventory."""


BYTECODE_NAME = re.compile(r"(?P<module>.+)\.(?:cpython-\d+|pypy\d+)\.(?:opt-\d+\.)?pyc\Z")
PYTEST_FILES = {
    ".pytest_cache/.gitignore", ".pytest_cache/CACHEDIR.TAG", ".pytest_cache/README.md",
    ".pytest_cache/v/cache/nodeids", ".pytest_cache/v/cache/lastfailed",
    ".pytest_cache/v/cache/stepwise",
}


def relative_path(value: str) -> PurePosixPath:
    """Reject noncanonical, absolute, traversal and platform-dependent rows."""
    path = PurePosixPath(value)
    if (not value or any(ord(c) < 32 for c in value) or "\\" in value
            or path.is_absolute() or PureWindowsPath(value).drive
            or any(part in {".", ".."} for part in value.split("/"))
            or value != path.as_posix() or path == PurePosixPath(".")):
        raise ManifestError(f"unsafe manifest path: {value!r}")
    return path


def regular_file(root: Path, relative: str) -> Path:
    """Validate every component without following links, before reading bytes."""
    parts = relative_path(relative).parts
    root = root.resolve(strict=True)
    path = root
    for index, part in enumerate(parts):
        path = path / part
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise ManifestError(f"missing or unreadable: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ManifestError(f"symlink forbidden: {relative}")
        expected = stat.S_ISREG(mode) if index == len(parts) - 1 else stat.S_ISDIR(mode)
        if not expected:
            raise ManifestError(f"non-regular file or non-directory parent: {relative}")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ManifestError(f"outside bundle root: {relative}")
    return path


def digest(path: Path) -> str:
    """Do not follow a final-component symlink or read a special file."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ManifestError(f"symlink or non-regular file: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ManifestError(f"non-regular file: {path.name}")
        h = hashlib.sha256()
        while block := handle.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def tracked_paths(root: Path) -> set[str]:
    """Tracked cache-looking files remain release content, even if ignored."""
    metadata = root / ".git"
    if not metadata.exists() and not metadata.is_symlink():
        return set()
    mode = metadata.lstat().st_mode
    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        raise ManifestError("symlink or non-regular Git metadata")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ManifestError("cannot determine tracked release files") from error
    return {name for name in result.stdout.decode("utf-8").split("\0") if name}


def generated_file(root: Path, relative: str, tracked: set[str]) -> bool:
    if relative in tracked:
        return False
    if relative.startswith("output/reproduction/"):
        return True
    if relative in PYTEST_FILES:
        return True
    path = PurePosixPath(relative)
    match = BYTECODE_NAME.fullmatch(path.name)
    if path.parent.name == "__pycache__" and match:
        # A plausible bytecode name without its actual source is not generated
        # cache state belonging to this bundle.
        source = path.parent.parent / (match["module"] + ".py")
        try:
            regular_file(root, source.as_posix())
        except ManifestError:
            return False
        return True
    return False


def inventory(root: Path, manifest_name: str = "MANIFEST.sha256") -> dict[str, Path]:
    root = root.resolve(strict=True)
    tracked = tracked_paths(root)
    files = {}

    def visit(folder: Path) -> None:
        for entry in sorted(folder.iterdir()):
            relative = entry.relative_to(root).as_posix()
            # Root Git metadata is not bundle content. Nested .git names are.
            if relative == ".git":
                if entry.is_symlink():
                    raise ManifestError("symlink forbidden: .git")
                continue
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ManifestError(f"symlink forbidden: {relative}")
            if stat.S_ISDIR(mode):
                visit(entry)
            elif not stat.S_ISREG(mode):
                raise ManifestError(f"non-regular file: {relative}")
            elif relative != manifest_name:
                path = regular_file(root, relative)
                if not generated_file(root, relative, tracked):
                    files[relative] = path

    visit(root)
    return files


def parse_manifest(text: str) -> dict[str, str]:
    rows = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ManifestError(f"empty manifest row: {number}")
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ManifestError(f"malformed manifest row: {number}")
        expected, relative = match.groups()
        relative_path(relative)
        if relative in rows:
            raise ManifestError(f"duplicate manifest entry: {relative}")
        rows[relative] = expected
    if not rows:
        raise ManifestError("empty manifest")
    return rows
