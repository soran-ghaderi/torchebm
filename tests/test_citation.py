r"""Consistency checks for the project citation.

``CITATION.cff`` is the single source of truth: GitHub renders it, and Zenodo
reads it to populate the archived record.  The BibTeX block is duplicated in
``README.md`` (GitHub and PyPI read the README raw, so it can never be
generated) and on the documentation landing page.  These tests pin the two
rendered copies to the metadata file so the duplication cannot drift.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

BIBTEX_KEY = "ghaderi2024torchebm"

CITATION_SURFACES = ("README.md", "docs/index.md")

_MISC_BLOCK = re.compile(r"```bibtex\n(@misc\{.*?\n\})\n```", re.S)
_FIELD = re.compile(r"^\s*(\w+)\s*=\s*\{(.*)\},?$", re.M)


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _bibtex_block(relative_path: str) -> str:
    blocks = _MISC_BLOCK.findall(_read(relative_path))
    assert len(blocks) == 1, f"{relative_path}: expected exactly one @misc block"
    return blocks[0]


def _bibtex_fields(block: str) -> dict:
    return {name: value for name, value in _FIELD.findall(block)}


@pytest.fixture(scope="module")
def cff() -> dict:
    return yaml.safe_load(_read("CITATION.cff"))


@pytest.fixture(scope="module")
def preferred(cff) -> dict:
    return cff["preferred-citation"]


def test_rendered_citations_are_identical():
    blocks = {path: _bibtex_block(path) for path in CITATION_SURFACES}
    assert len(set(blocks.values())) == 1, f"citation blocks differ: {list(blocks)}"


def test_citation_is_not_duplicated_further():
    for path in ROOT.joinpath("docs").rglob("*.md"):
        if path.relative_to(ROOT).as_posix() in CITATION_SURFACES:
            continue
        assert BIBTEX_KEY not in path.read_text(encoding="utf-8"), (
            f"{path.relative_to(ROOT)} carries its own copy of the citation; "
            f"link to the canonical one in {CITATION_SURFACES[1]} instead"
        )


@pytest.mark.parametrize("path", CITATION_SURFACES)
def test_bibtex_matches_citation_cff(path, cff, preferred):
    fields = _bibtex_fields(_bibtex_block(path))

    assert _bibtex_block(path).startswith(f"@misc{{{BIBTEX_KEY},")

    author = preferred["authors"][0]
    assert fields["author"] == f"{author['family-names']}, {author['given-names']}"

    assert fields["title"].replace("{", "").replace("}", "") == preferred["title"]
    assert preferred["title"] == cff["title"]

    assert fields["year"] == str(preferred["year"])

    url = re.fullmatch(r"\\url\{(.*)\}", fields["howpublished"])
    assert url is not None, "howpublished must wrap the url in \\url{...}"
    assert url.group(1) == preferred["url"]


def test_bibtex_key_encodes_author_and_year(preferred):
    author = preferred["authors"][0]["family-names"].lower()
    assert BIBTEX_KEY == f"{author}{preferred['year']}torchebm"
