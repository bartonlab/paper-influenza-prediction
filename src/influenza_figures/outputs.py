"""Collision-safe, registry-owned figure output transactions."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .paths import ProjectPaths
from .registry import FIGURE_REGISTRY, FigureRegistry


class OutputOwnershipError(ValueError):
    """Raised when a caller requests an output owned by another recipe."""


class OutputExistsError(FileExistsError):
    """Raised when a render would replace a file without explicit permission."""


class OutputBusyError(RuntimeError):
    """Raised when another compliant render currently holds the output lock."""


class OutputNotProducedError(RuntimeError):
    """Raised when a renderer exits without creating its reserved temporary file."""


class OutputManager:
    """Resolve and atomically commit files owned by a ``FigureRegistry``.

    The manager never overwrites by default.  A render writes to a temporary
    file in the target directory and commits it atomically.  A hidden lock file
    prevents two compliant processes from rendering the same figure at once.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        registry: FigureRegistry = FIGURE_REGISTRY,
    ) -> None:
        self.paths = paths
        self.registry = registry

    def output_path(self, owner_key: str) -> Path:
        spec = self.registry.get(owner_key)
        return self.paths.figure_path(spec.output_name)

    def assert_owner(self, owner_key: str, output_name: str) -> None:
        registered_owner = self.registry.owner_of(output_name)
        if registered_owner != owner_key:
            raise OutputOwnershipError(
                f"{output_name!r} is owned by {registered_owner!r}, not {owner_key!r}"
            )

    @contextmanager
    def reserve(self, owner_key: str, *, overwrite: bool = False) -> Iterator[Path]:
        """Yield a temporary output path and atomically commit on success.

        The yielded path retains the target extension so libraries such as
        Matplotlib can infer the desired file format.
        """

        spec = self.registry.get(owner_key)
        target = self.paths.figure_path(spec.output_name)
        target.parent.mkdir(parents=True, exist_ok=True)

        lock_directory = self.paths.figure_path(".locks")
        lock_directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(spec.output_name.casefold().encode("utf-8")).hexdigest()
        lock_path = lock_directory / f"{digest}.lock"

        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise OutputBusyError(
                f"output {target.name!r} is already reserved; lock: {lock_path}"
            ) from exc

        temporary = target.with_name(
            f".{target.stem}.{uuid4().hex}{target.suffix}"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
                lock_file.write(f"owner={owner_key}\ntarget={target.name}\npid={os.getpid()}\n")

            if target.exists() and not overwrite:
                raise OutputExistsError(
                    f"refusing to overwrite existing output {target}; pass overwrite=True explicitly"
                )

            yield temporary

            if not temporary.is_file():
                raise OutputNotProducedError(
                    f"renderer did not create reserved output {temporary}"
                )

            if overwrite:
                os.replace(temporary, target)
            else:
                # A hard-link commit is atomic and fails rather than replacing a
                # target created by a non-cooperating process during the render.
                try:
                    os.link(temporary, target)
                except FileExistsError as exc:
                    raise OutputExistsError(
                        f"output appeared during render and was not overwritten: {target}"
                    ) from exc
                temporary.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()
            if lock_path.exists():
                lock_path.unlink()

    def write_bytes(self, owner_key: str, content: bytes, *, overwrite: bool = False) -> Path:
        """Small convenience used by tests and non-Matplotlib renderers."""

        with self.reserve(owner_key, overwrite=overwrite) as temporary:
            temporary.write_bytes(content)
        return self.output_path(owner_key)
