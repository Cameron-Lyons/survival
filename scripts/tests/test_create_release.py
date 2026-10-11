"""Release retries must preserve the selected commit and resume publishing."""

# ruff: noqa: PT009, PT027

from __future__ import annotations

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("create_release", ROOT / "scripts/create_release.py")
assert SPEC is not None  # noqa: S101
assert SPEC.loader is not None  # noqa: S101
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
TAG = "v2.0.1"


class CreateReleaseTests(unittest.TestCase):
    def run_release(
        self,
        *,
        tag_commit: str | None = None,
        published_commit: str = COMMIT,
        release_status: int = 200,
        has_release: bool = False,
        draft: bool = False,
        selected_commit: str = COMMIT,
        python_version: str = "2.0.1",
    ) -> list[tuple[str, ...]]:
        commands = []
        self.commands = commands
        fetched = False

        def run(*args, check=True):
            nonlocal fetched
            commands.append(args)
            status = 0
            stdout = ""
            stderr = ""
            if args == ("git", "rev-parse", "HEAD"):
                stdout = COMMIT
            elif args[:3] == ("git", "rev-parse", "--verify"):
                target = published_commit if fetched else tag_commit
                stdout = target or ""
                status = 0 if target else 1
            elif args[:2] == ("git", "fetch"):
                fetched = True
            elif args[:2] == ("gh", "api"):
                pages = [[{"tag_name": "v1.0.0", "draft": False}]]
                pages.append([{"tag_name": TAG, "draft": draft}] if has_release else [])
                stdout = json.dumps(pages)
                status = 0 if release_status == 200 else 1
                stderr = f"gh: request failed (HTTP {release_status})" if status else ""
            result = subprocess.CompletedProcess(args, status, stdout, stderr)
            if check and status:
                raise subprocess.CalledProcessError(status, args, stdout, stderr)
            return result

        def read_text(path):
            if path.name == "Cargo.toml":
                return '[package]\nversion = "2.0.1"\n'
            return f'[project]\nversion = "{python_version}"\n'

        with (
            patch.object(release, "_run", side_effect=run),
            patch.object(Path, "read_text", read_text),
            patch.dict("os.environ", {"GITHUB_SHA": selected_commit, "GH_REPO": "owner/survival"}),
        ):
            release.create_release()
        return commands

    def assert_dispatch(self, commands):
        self.assertEqual(
            commands[-1],
            (
                "gh",
                "workflow",
                "run",
                "publish.yml",
                "--ref",
                TAG,
                "--raw-field",
                "target=all",
                "--raw-field",
                f"ref={COMMIT}",
            ),
        )

    def test_new_release_targets_selected_commit_and_dispatches_both_registries(self):
        commands = self.run_release()
        self.assertIn(
            (
                "gh",
                "release",
                "create",
                TAG,
                "--target",
                COMMIT,
                "--title",
                TAG,
                "--generate-notes",
            ),
            commands,
        )
        self.assert_dispatch(commands)

    def test_existing_matching_tag_without_release_is_completed(self):
        commands = self.run_release(tag_commit=COMMIT)
        self.assertTrue(any(command[:3] == ("gh", "release", "create") for command in commands))
        self.assert_dispatch(commands)

    def test_existing_published_release_retries_dispatch_without_recreating(self):
        commands = self.run_release(tag_commit=COMMIT, has_release=True)
        self.assertFalse(any(command[:2] == ("gh", "release") for command in commands))
        self.assert_dispatch(commands)

    def test_existing_draft_is_published_before_dispatch(self):
        commands = self.run_release(tag_commit=COMMIT, has_release=True, draft=True)
        self.assertIn(("gh", "release", "edit", TAG, "--target", COMMIT, "--draft=false"), commands)
        self.assert_dispatch(commands)

    def test_existing_draft_without_tag_is_pinned_to_selected_commit(self):
        commands = self.run_release(has_release=True, draft=True)
        self.assertIn(("gh", "release", "edit", TAG, "--target", COMMIT, "--draft=false"), commands)
        self.assertFalse(any(command[:3] == ("gh", "release", "create") for command in commands))
        self.assert_dispatch(commands)

    def test_existing_tag_cannot_publish_different_selected_commit(self):
        with self.assertRaisesRegex(ValueError, "Existing tag"):
            self.run_release(tag_commit=OTHER_COMMIT)
        self.assertFalse(any(command[0] == "gh" for command in self.commands))

    def test_remote_tag_is_rechecked_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, "Published tag"):
            self.run_release(published_commit=OTHER_COMMIT)
        self.assertFalse(any(command[:2] == ("gh", "workflow") for command in self.commands))

    def test_api_failure_is_not_treated_as_a_missing_release(self):
        for status in (403, 404, 503):
            with self.subTest(status=status), self.assertRaises(subprocess.CalledProcessError):
                self.run_release(release_status=status)
            self.assertFalse(any(command[:2] == ("gh", "release") for command in self.commands))
            self.assertFalse(any(command[:2] == ("gh", "workflow") for command in self.commands))

    def test_checkout_must_match_workflow_selected_commit(self):
        with self.assertRaisesRegex(ValueError, "Checkout differs"):
            self.run_release(selected_commit=OTHER_COMMIT)
        self.assertFalse(any(command[0] == "gh" for command in self.commands))

    def test_cargo_and_python_versions_must_match(self):
        with self.assertRaisesRegex(ValueError, "differs from Python"):
            self.run_release(python_version="2.0.0")
        self.assertFalse(any(command[0] == "gh" for command in self.commands))


if __name__ == "__main__":
    unittest.main()
