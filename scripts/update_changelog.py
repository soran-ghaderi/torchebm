r"""Maintain CHANGELOG.md from rendered git-cliff bodies.

Bodies come from git-cliff (see cliff.toml). CI runs this on the PR branch
(.github/workflows/changelog.yml): it strips sections newer than the last
GitHub Release (only the workflow itself writes those), then promotes the
[Unreleased] block to the version the merge will mint. Sections at or below
the last release are never touched, so manual backfills survive.

Usage:
    python scripts/update_changelog.py unreleased --body FILE
        Replace the [Unreleased] body.

    python scripts/update_changelog.py release --version vX.Y.Z \
        --date YYYY-MM-DD --body FILE
        Replace the [Unreleased] block with a version section and open a
        fresh empty [Unreleased] above it.

    python scripts/update_changelog.py strip --above vX.Y.Z
        Remove every version section newer than vX.Y.Z.
"""

import argparse
import pathlib
import re
import sys

CHANGELOG = pathlib.Path(__file__).resolve().parent.parent / "CHANGELOG.md"
HEADING = "## [Unreleased]"


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.lstrip("v").split("."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("unreleased", "release", "strip"))
    parser.add_argument("--body", help="file with the rendered body")
    parser.add_argument("--version", help="release mode: version (v prefix ok)")
    parser.add_argument("--date", help="release mode: YYYY-MM-DD")
    parser.add_argument("--above", help="strip mode: last released version")
    args = parser.parse_args()
    if args.mode == "release" and not (args.version and args.date and args.body):
        parser.error("release mode requires --version, --date and --body")
    if args.mode == "unreleased" and not args.body:
        parser.error("unreleased mode requires --body")
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

    body = pathlib.Path(args.body).read_text().strip()
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
