r"""Splice a rendered changes body into CHANGELOG.md.

Only the ``## [Unreleased]`` block is ever touched; released sections below
it are frozen. Bodies come from git-cliff (see cliff.toml). Run locally on
the PR branch before merging to master; the changelog change ships in the PR
itself (the master ruleset rejects direct pushes). Render bodies from
``<last-release-tag>..origin/master`` so bullets link canonical master SHAs:

    git fetch origin master
    last=$(gh release list --limit 1 --json tagName --jq '.[0].tagName')
    git-cliff --config cliff.toml --strip all "${last}..origin/master" -o body.md

Usage:
    python scripts/update_changelog.py unreleased --body FILE
        Replace the [Unreleased] body.

    python scripts/update_changelog.py release --version vX.Y.Z \
        --date YYYY-MM-DD --body FILE
        Replace the [Unreleased] block with a version section and open a
        fresh empty [Unreleased] above it.
"""

import argparse
import pathlib
import re
import sys

CHANGELOG = pathlib.Path(__file__).resolve().parent.parent / "CHANGELOG.md"
HEADING = "## [Unreleased]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("unreleased", "release"))
    parser.add_argument("--body", required=True, help="file with the rendered body")
    parser.add_argument("--version", help="release mode: version (v prefix ok)")
    parser.add_argument("--date", help="release mode: YYYY-MM-DD")
    args = parser.parse_args()
    if args.mode == "release" and not (args.version and args.date):
        parser.error("release mode requires --version and --date")

    body = pathlib.Path(args.body).read_text().strip()
    text = CHANGELOG.read_text()
    if HEADING not in text:
        sys.exit(f"CHANGELOG.md has no '{HEADING}' section")
    head, rest = text.split(HEADING, 1)
    match = re.search(r"\n## \[", rest)
    frozen = rest[match.start() :] if match else "\n"

    if args.mode == "unreleased":
        new = f"{head}{HEADING}\n\n{body}\n{frozen}"
    else:
        version = args.version.lstrip("v")
        new = (
            f"{head}{HEADING}\n\n"
            f"## [{version}] - {args.date}\n\n{body}\n{frozen}"
        )
    CHANGELOG.write_text(new)


if __name__ == "__main__":
    main()
