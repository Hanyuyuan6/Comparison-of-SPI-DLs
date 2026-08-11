"""Small path guards for user-provided run identifiers."""

import re
from pathlib import Path


_RUN_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)


def safe_child_directory(root, name, *, field_name="experiment_name"):
    """Return ``root/name`` after proving that *name* is one path component.

    Root directories are intentionally configurable local paths.  The run name,
    however, must never replace or escape that configured root.
    """
    if not isinstance(name, str):
        raise TypeError(f"{field_name} must be a string")
    if _RUN_NAME_RE.fullmatch(name) is None or name in {".", ".."}:
        raise ValueError(
            f"{field_name} must be 1-128 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return Path(root) / name
