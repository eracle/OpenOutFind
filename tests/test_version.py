# tests/test_version.py
"""What build am I — against a real ``.git`` on disk, built without a git binary.

The sha path reads plain text files (``HEAD``, a loose ref, ``packed-refs``, the
``gitdir:`` pointer), so the fixtures below write exactly those bytes rather than
shelling out to ``git init``. That is not mocking: the files are the real on-disk
format, and the parser under test does the parsing. It is only the *construction*
that stops needing a binary the runtime image doesn't ship.

The two functions that do shell out — ``_commit_date`` and ``_working_copy_dirty``
— are documented to degrade to ``UNKNOWN``/``None`` when ``git`` is missing, so
they are exercised by stubbing ``_git``, which is the seam that exists for it.
"""

import pytest

from openoutfind.core import version

SHA = "947927d3f0c4b1e6a8d2f5b9c7e1a4d8f2b6c0e3"
AUTHORED_AT = "1786096800"  # 2026-08-07T10:00:00Z


def _git_says(*, date=AUTHORED_AT, status=""):
    """Stub for ``version._git``, dispatching on the subcommand.

    One lambda answering every call with the same string made ``git status`` look
    non-empty, i.e. every checkout dirty.
    """
    return lambda *a: date if a[0] == "show" else status


def _checkout(root, head: str, refs: dict | None = None, packed: str = "") -> None:
    """Write a ``.git`` directory: ``HEAD``, any loose refs, and ``packed-refs``."""
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text(head)
    for ref, sha in (refs or {}).items():
        (git_dir / ref).write_text(f"{sha}\n")
    if packed:
        (git_dir / "packed-refs").write_text(packed)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Point the module at a scratch checkout and clear its per-process cache."""
    monkeypatch.setattr(version, "REPO_ROOT", tmp_path)
    version._build.cache_clear()
    yield tmp_path
    version._build.cache_clear()


# ── identity ─────────────────────────────────────────────────────────

def test_reads_head_sha_from_a_loose_ref(_isolated):
    _checkout(_isolated, "ref: refs/heads/main\n", {"refs/heads/main": SHA})
    assert version.commit_sha() == SHA


def test_resolves_head_from_packed_refs(_isolated):
    """A fresh clone keeps refs packed, so the loose ref file simply isn't there."""
    _checkout(_isolated, "ref: refs/heads/main\n",
              packed=f"# pack-refs with: peeled fully-peeled sorted \n{SHA} refs/heads/main\n")
    assert version.commit_sha() == SHA


def test_packed_refs_skips_comments_and_peeled_lines(_isolated):
    """``^`` lines carry the tag's target and must never be read as a ref's sha."""
    _checkout(_isolated, "ref: refs/heads/main\n",
              packed=("# pack-refs with: peeled \n"
                      f"{'b' * 40} refs/tags/v1\n"
                      f"^{'c' * 40}\n"
                      f"{SHA} refs/heads/main\n"))
    assert version.commit_sha() == SHA


def test_resolves_detached_head(_isolated):
    _checkout(_isolated, f"{SHA}\n")
    assert version.commit_sha() == SHA


def test_resolves_gitdir_pointer_file(_isolated, tmp_path, monkeypatch):
    """Development happens in a submodule, where .git is a file, not a directory."""
    real = tmp_path / "real"
    real.mkdir()
    _checkout(real, "ref: refs/heads/main\n", {"refs/heads/main": SHA})

    checkout = tmp_path / "elsewhere"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {real / '.git'}\n")
    monkeypatch.setattr(version, "REPO_ROOT", checkout)
    version._build.cache_clear()

    assert version.commit_sha() == SHA


def test_missing_git_metadata_is_unknown_not_a_crash(_isolated):
    assert version.commit_sha() == version.UNKNOWN
    assert version.calver() == version.UNKNOWN


# ── ordering ─────────────────────────────────────────────────────────

def test_calver_comes_from_the_authored_date(_isolated, monkeypatch):
    """``%at``, not ``%ct``: a rebase must not move the version."""
    _checkout(_isolated, f"{SHA}\n")
    monkeypatch.setattr(version, "_git", _git_says())
    version._build.cache_clear()
    assert version.calver() == "2026.08.07"


def test_calver_is_unknown_without_a_git_binary(_isolated, monkeypatch):
    """The image ships no ``git``, and the hub resolves the date from the sha anyway."""
    _checkout(_isolated, f"{SHA}\n")
    monkeypatch.setattr(version, "_git", lambda *a: None)
    version._build.cache_clear()
    assert version.commit_sha() == SHA
    assert version.calver() == version.UNKNOWN


def test_version_string_pairs_the_date_with_the_sha(_isolated, monkeypatch):
    _checkout(_isolated, f"{SHA}\n")
    monkeypatch.setattr(version, "_git", _git_says())
    version._build.cache_clear()
    assert version.version_string() == f"2026.08.07+g{SHA[:7]}"
    assert version.user_agent() == f"OpenOutFind/2026.08.07+g{SHA[:7]}"


# ── local modification ───────────────────────────────────────────────

def test_edited_working_copy_reports_dirty(_isolated, monkeypatch):
    _checkout(_isolated, f"{SHA}\n")
    monkeypatch.setattr(version, "_git", _git_says(status=" M app.py"))
    version._build.cache_clear()
    assert version.is_dirty() is True
    assert version.version_string().endswith(".dirty")


def test_clean_working_copy_is_not_dirty(_isolated, monkeypatch):
    _checkout(_isolated, f"{SHA}\n")
    monkeypatch.setattr(version, "_git", _git_says())
    version._build.cache_clear()
    assert version.is_dirty() is False
    assert ".dirty" not in version.version_string()


def test_undeterminable_dirtiness_is_none_not_false(_isolated, monkeypatch):
    """No git binary must not produce a confident 'clean' we never verified."""
    _checkout(_isolated, f"{SHA}\n")
    monkeypatch.setattr(version, "_git", lambda *a: None)
    version._build.cache_clear()
    assert version.is_dirty() is None
    assert ".dirty" not in version.version_string()


# ── build override ───────────────────────────────────────────────────

def test_build_env_var_wins_over_the_checkout(_isolated, monkeypatch):
    """The image is built without a ``.git``, so the env var is how it knows itself."""
    _checkout(_isolated, f"{SHA}\n")
    monkeypatch.setenv("OPENOUTFIND_BUILD", f"{'a' * 40}@2026.01.01")
    version._build.cache_clear()
    assert version.commit_sha() == "a" * 40
    assert version.calver() == "2026.01.01"
