"""What build of OpenOutFind is this instance running.

**This is the build, not the package version.** The two answer different questions
and neither replaces the other: PyPI carries a released number (`0.1.<commits since
v0.1.0>`, derived at publish time), while a server deploys by ``git clone`` off
``main`` and has no released number at all. The deployable unit there is the commit,
so the commit *is* the version. This module names it two ways:

* ``commit_sha`` — the identity. Unique, immutable, and reachable: from it the
  hub can look the build up in the published history.
* ``calver`` — the ordering. ``YYYY.MM.DD`` from the commit's own date, because
  hashes don't compare. Two commits on one day share a date; the sha separates
  them.

``is_dirty`` covers the third case a version has to admit to: a working copy with
uncommitted edits is a build nobody can reconstruct, so claiming to be the commit
would be a false claim. ``None`` means "couldn't tell" — never a hopeful ``False``.

Read straight from ``.git`` (no ``git`` binary needed for the sha, no GitPython)
because the daemon runs from a bind-mounted checkout. ``OPENOUTFIND_BUILD`` wins
when set, for images built without a ``.git`` alongside.
"""
from __future__ import annotations

import functools
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root: openoutfind/core/version.py → up two.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UNKNOWN = "unknown"
_SHA_DISPLAY_LEN = 7
_GIT_TIMEOUT_S = 5


def commit_sha() -> str:
    """The full 40-char commit sha of this checkout, or ``"unknown"``."""
    return _build()["commit_sha"]


def calver() -> str:
    """``YYYY.MM.DD`` of the commit's authored date, or ``"unknown"``."""
    return _build()["calver"]


def is_dirty() -> bool | None:
    """Whether the working copy has uncommitted changes (``None`` = undetermined)."""
    return _build()["is_dirty"]


def version_string() -> str:
    """The human form: ``2026.08.07+g947927d``, with ``.dirty`` when it applies."""
    build = _build()
    sha = build["commit_sha"]
    short = sha[:_SHA_DISPLAY_LEN] if sha != UNKNOWN else UNKNOWN
    suffix = ".dirty" if build["is_dirty"] else ""
    return f"{build['calver']}+g{short}{suffix}"


def user_agent() -> str:
    """The product token sent on every hub call (RFC 9110 form)."""
    return f"OpenOutFind/{version_string()}"


@functools.lru_cache(maxsize=1)
def _build() -> dict:
    """Resolve the build once per process — the checkout can't change under us."""
    override = os.environ.get("OPENOUTFIND_BUILD")
    if override:
        sha, _, date = override.partition("@")
        return {"commit_sha": sha or UNKNOWN, "calver": date or UNKNOWN, "is_dirty": None}
    sha = _read_head_sha()
    return {
        "commit_sha": sha,
        "calver": _commit_date(sha),
        "is_dirty": _working_copy_dirty(),
    }


def _git_dir() -> Path:
    """The real git directory for this checkout.

    Usually ``<repo>/.git``, but in a submodule or a linked worktree ``.git`` is a
    *file* holding ``gitdir: <path>`` that points at the true directory (relative
    to the repo root). Development happens in a submodule, production in a plain
    clone — both have to work.
    """
    dot_git = REPO_ROOT / ".git"
    if dot_git.is_dir():
        return dot_git
    pointer = _read_text(dot_git) or ""
    if pointer.startswith("gitdir:"):
        target = Path(pointer.removeprefix("gitdir:").strip())
        return target if target.is_absolute() else (REPO_ROOT / target).resolve()
    return dot_git


def _read_head_sha() -> str:
    """HEAD's sha by reading ``.git`` directly — no ``git`` binary required.

    Handles the two shapes HEAD takes: a detached sha, or a ref pointer whose
    target lives in ``refs/`` (loose) or in ``packed-refs`` (after a gc/clone).
    """
    git_dir = _git_dir()
    head = _read_text(git_dir / "HEAD")
    if not head:
        return UNKNOWN
    if not head.startswith("ref:"):
        return head  # detached HEAD holds the sha itself
    ref = head.removeprefix("ref:").strip()
    return _read_text(git_dir / ref) or _packed_ref(git_dir, ref) or UNKNOWN


def _packed_ref(git_dir: Path, ref: str) -> str | None:
    """Look ``ref`` up in ``packed-refs`` (where a fresh clone keeps its refs)."""
    for line in (_read_text(git_dir / "packed-refs") or "").splitlines():
        if line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref:
            return sha
    return None


def _read_text(path: Path) -> str | None:
    """File contents, stripped — ``None`` if it isn't there or isn't readable."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _commit_date(sha: str) -> str:
    """The commit's authored date as ``YYYY.MM.DD``.

    Parsed out of the commit object via ``git``; without it (or without the object)
    the date is simply unknown — the hub can still resolve the date from the sha
    against the published history, so this is a convenience, not the source of truth.

    ``%at`` (author date), not ``%ct``: a rebase or a cherry-pick rewrites the
    committer timestamp, so the same code would report a different CalVer depending
    on how it reached the branch. The author date is when the change was written,
    which is what a build identifier should name.
    """
    if sha == UNKNOWN:
        return UNKNOWN
    stamp = _git("show", "-s", "--format=%at", sha)
    if not stamp or not stamp.isdigit():
        return UNKNOWN
    return datetime.fromtimestamp(int(stamp), tz=UTC).strftime("%Y.%m.%d")


def _working_copy_dirty() -> bool | None:
    """``True`` if tracked files differ from HEAD, ``None`` if ``git`` can't say.

    Deliberately not reimplemented over the git index: getting it subtly wrong
    would produce confident-but-false "clean" answers, and an honest ``None``
    beats that.
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None
    return bool(status)


def _git(*args: str) -> str | None:
    """Run a read-only ``git`` command in the repo; ``None`` on any failure.

    Failure is expected, not exceptional: the runtime image may not ship ``git``,
    and a checkout owned by another uid trips git's ``dubious ownership`` guard.
    """
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell, no user input
            ["git", "-C", str(REPO_ROOT), *args],  # noqa: S607
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("version: git %s unavailable: %s", args[0], exc)
        return None
    if result.returncode != 0:
        logger.debug("version: git %s failed: %s", args[0], result.stderr.strip())
        return None
    return result.stdout.strip()
