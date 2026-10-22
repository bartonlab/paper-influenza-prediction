"""Repository-relative paths with containment checks and no cwd assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ProjectRootNotFoundError(FileNotFoundError):
    """Raised when repository discovery cannot find a project marker."""


class PathOutsideRootError(ValueError):
    """Raised when an absolute or traversing relative path escapes its root."""


def _resolve_under(base: Path, relative: str | Path) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute():
        raise PathOutsideRootError(f"absolute paths are not allowed: {relative!r}")

    base_resolved = base.resolve()
    candidate = (base_resolved / candidate_relative).resolve(strict=False)
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise PathOutsideRootError(
            f"path {relative!r} escapes configured root {base_resolved}"
        ) from exc
    return candidate


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """All paths used by plotting, rooted explicitly at one repository."""

    root: Path
    data: Path
    figures: Path
    archive: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            data=resolved / "data" / "plotting",
            figures=resolved / "figures",
            archive=resolved / "archive",
        )

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "ProjectPaths":
        """Walk upward to a Git or Python-project marker.

        Discovery is an entry-point convenience; library and test code should
        prefer ``from_root`` so the selected repository is unambiguous.
        """

        location = Path.cwd() if start is None else Path(start)
        location = location.expanduser().resolve()
        if location.is_file():
            location = location.parent

        for candidate in (location, *location.parents):
            if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
                return cls.from_root(candidate)
        raise ProjectRootNotFoundError(f"no project root found above {location}")

    def data_path(self, relative: str | Path, *, must_exist: bool = False) -> Path:
        path = _resolve_under(self.data, relative)
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path

    def figure_path(self, relative: str | Path) -> Path:
        return _resolve_under(self.figures, relative)

    def archive_path(self, relative: str | Path) -> Path:
        return _resolve_under(self.archive, relative)

