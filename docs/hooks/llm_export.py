"""MkDocs hook to generate additional LLM-friendly documentation files."""

import os
import logging
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.llm_export")


def on_post_build(config, **kwargs):
    """Generate additional LLM context files after build."""
    site_dir = Path(config["site_dir"])
    docs_dir = Path(config["docs_dir"])
    
    # The mkdocs-llmstxt plugin handles llms.txt and llms-full.txt generation
    # This hook can be extended for additional custom LLM exports if needed
    
    llms_txt = site_dir / "llms.txt"
    llms_full = site_dir / "llms-full.txt"
    
    if llms_txt.exists():
        log.info(f"llms.txt generated at: {llms_txt}")
    
    if llms_full.exists():
        log.info(f"llms-full.txt generated at: {llms_full}")
