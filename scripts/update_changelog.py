r"""Maintain CHANGELOG.md from rendered git-cliff bodies.

Bodies come from git-cliff (see cliff.toml). CI runs this on the PR branch
(.github/workflows/changelog.yml): it strips sections newer than the last
GitHub Release (only the workflow itself writes those), then inserts a
section for the version the merge will mint at the top of the file. Every
merge to master is released, so the file holds released sections only (no
[Unreleased] block); sections at or below the last release are never
touched, so manual backfills survive.

Usage:
    python scripts/update_changelog.py release --version vX.Y.Z \
        --date YYYY-MM-DD --body FILE [--link URL]
        Insert a version section at the top (replacing any [Unreleased]
        block a historical file still carries). With --link the heading
        becomes an inline link, e.g. to the release's compare view.

    python scripts/update_changelog.py strip --above vX.Y.Z
        Remove every version section newer than vX.Y.Z.
"""

import argparse
import pathlib
import re

CHANGELOG = pathlib.Path(__file__).resolve().parent.parent / "CHANGELOG.md"
HEADING = "## [Unreleased]"


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.lstrip("v").split("."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("release", "strip"))
    parser.add_argument("--body", help="release mode: file with the rendered body")
    parser.add_argument("--version", help="release mode: version (v prefix ok)")
    parser.add_argument("--date", help="release mode: YYYY-MM-DD")
    parser.add_argument("--above", help="strip mode: last released version")
    parser.add_argument("--link", help="release mode: URL for the heading link")
    args = parser.parse_args()
    if args.mode == "release" and not (args.version and args.date and args.body):
        parser.error("release mode requires --version, --date and --body")
    if args.mode == "strip" and not args.above:
        parser.error("strip mode requires --above")

    text = CHANGELOG.read_text()

    if args.mode == "strip":
        floor = _version_tuple(args.above)
        head, *sections = re.split(r"(?m)^(?=## \[)", text)
        kept = [head]
        for section in sections:
            match = re.match(r"## \[(\d+(?:\.\d+)*)\]", section)
            if match and _version_tuple(match.group(1)) > floor:
                continue
            kept.append(section)
        CHANGELOG.write_text("".join(kept))
        return

    if HEADING in text:
        head, rest = text.split(HEADING, 1)
        match = re.search(r"(?m)^## \[", rest)
        text = head + (rest[match.start() :] if match else "")

    body = pathlib.Path(args.body).read_text().strip()
    version = args.version.lstrip("v")
    title = f"[{version}]({args.link})" if args.link else f"[{version}]"
    section = f"## {title} - {args.date}\n\n{body}\n\n"
    match = re.search(r"(?m)^## \[", text)
    insert_at = match.start() if match else len(text)
    CHANGELOG.write_text(text[:insert_at] + section + text[insert_at:])


if __name__ == "__main__":
    main()
