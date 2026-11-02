r"""MkDocs hook for resolving relative links and enriching GitHub URLs.

- Relative file links pointing outside ``docs/`` become GitHub blob/tree URLs.
- Raw GitHub issue/PR/project links get descriptive titles and an icon.

The target repo is taken from ``repo_url`` in mkdocs config, so this stays
correct if the project is renamed or forked.
"""

import re
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

ROOT_DIR = Path(__file__).resolve().parents[2]
DOC_DIR = ROOT_DIR / "docs"

DEFAULT_BRANCH = "master"

gh_icon = ":octicons-mark-github-16:"

TITLE = r"(?P<title>[^\[\]<>]+?)"
REPO = r"(?P<repo>.+?/.+?)"
TYPE = r"(?P<type>issues|pull|projects)"
NUMBER = r"(?P<number>\d+)"
PATH = r"(?P<path>[^\s]+?)"
FRAGMENT = r"(?P<fragment>#[^\s]+)?"
URL = f"https://github.com/{REPO}/{TYPE}/{NUMBER}{FRAGMENT}"
RELATIVE = rf"(?!(https?|ftp)://|#){PATH}{FRAGMENT}"

TITLES = {"issues": "Issue ", "pull": "Pull Request ", "projects": "Project "}

github_link = re.compile(rf"(\[{TITLE}\]\(|<){URL}(\)|>)")
relative_link = re.compile(rf"\[{TITLE}\]\({RELATIVE}\)")


def _repo_slug(config: MkDocsConfig) -> str:
    repo_url = str(config.get("repo_url") or "").rstrip("/")
    return repo_url.split("github.com/")[-1]


def on_page_markdown(markdown: str, *, page: Page, config: MkDocsConfig, files: Files) -> str:
    repo_slug = _repo_slug(config)

    def replace_relative_link(match: re.Match) -> str:
        title = match.group("title")
        path_str = match.group("path")
        resolved = (Path(page.file.abs_src_path).parent / path_str).resolve()
        fragment = match.group("fragment") or ""

        if not resolved.exists() or resolved.is_relative_to(DOC_DIR):
            return match.group(0)

        slug = "tree" if resolved.is_dir() else "blob"
        rel = resolved.relative_to(ROOT_DIR)
        url = f"https://github.com/{repo_slug}/{slug}/{DEFAULT_BRANCH}/{rel}{fragment}"
        return f"[{gh_icon} {title}]({url})"

    def replace_github_link(match: re.Match) -> str:
        repo = match.group("repo")
        link_type = match.group("type")
        number = match.group("number")
        title = match.group("title") or ""
        fragment = match.group("fragment") or ""

        if not title:
            title = TITLES[link_type]
            if repo_slug.split("/")[0] not in repo:
                title += repo
            title += f"#{number}"

        url = f"https://github.com/{repo}/{link_type}/{number}{fragment}"
        return f"[{gh_icon} {title}]({url})"

    # Generated (in-memory) pages have no abs_src_path and already had their
    # relative links resolved at generation time; only enrich GitHub links.
    if page.file.abs_src_path is not None:
        markdown = relative_link.sub(replace_relative_link, markdown)
    markdown = github_link.sub(replace_github_link, markdown)
    return markdown
