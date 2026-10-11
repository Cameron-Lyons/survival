"""Create or resume a release of this checkout, then dispatch both registries."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True)  # noqa: S603


def _tag_commit(tag: str) -> str | None:
    result = _run("git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def create_release() -> None:
    version = tomllib.loads(Path("Cargo.toml").read_text())["package"]["version"]
    python_version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    if version != python_version:
        raise ValueError(f"Cargo version {version} differs from Python version {python_version}")
    tag = f"v{version}"
    commit = _run("git", "rev-parse", "HEAD").stdout.strip()
    if commit != os.environ["GITHUB_SHA"]:
        raise ValueError("Checkout differs from the commit selected for this Release run")
    tagged_commit = _tag_commit(tag)
    if tagged_commit is not None and tagged_commit != commit:
        raise ValueError(f"Existing tag {tag} points to {tagged_commit}, not {commit}")

    repository = os.environ["GH_REPO"]
    # The by-tag API only promises published releases. Authenticated listings
    # also include drafts, so a draft retry must search the listing instead.
    result = _run("gh", "api", "--paginate", "--slurp", f"repos/{repository}/releases?per_page=100")
    existing_release = next(
        (item for page in json.loads(result.stdout) for item in page if item["tag_name"] == tag),
        None,
    )
    if existing_release is None:
        _run("gh", "release", "create", tag, "--target", commit, "--title", tag, "--generate-notes")
    elif existing_release["draft"]:
        _run("gh", "release", "edit", tag, "--target", commit, "--draft=false")

    # Release creation can create the tag remotely. Check it again before
    # dispatching, including when resuming an already published release.
    _run("git", "fetch", "--no-tags", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
    if _tag_commit(tag) != commit:
        raise ValueError(f"Published tag {tag} does not point to {commit}")
    _run(
        "gh",
        "workflow",
        "run",
        "publish.yml",
        "--ref",
        tag,
        "--raw-field",
        "target=all",
        "--raw-field",
        f"ref={commit}",
    )
    print(f"Publishing {tag} from {commit} to PyPI and crates.io")


if __name__ == "__main__":
    create_release()
