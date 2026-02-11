#!/usr/bin/env python3
r"""Generate LLM-friendly API documentation for TorchEBM.

This script creates compact, LLM-optimized documentation files:
- docs/llm/all.md: Complete API reference with signatures and docstrings
- docs/llm/llms-ctx.txt: Context file for direct LLM consumption

Usage:
    python generate_llm_docs.py
"""

import inspect
import importlib
import pkgutil
from pathlib import Path
from typing import Any


def get_signature(obj: Any) -> str:
    """Get function/method signature as string."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "()"


def get_first_docstring_line(obj: Any) -> str:
    """Extract first line of docstring."""
    doc = inspect.getdoc(obj)
    if not doc:
        return ""
    return doc.split("\n")[0].strip()


def get_full_docstring(obj: Any) -> str:
    """Get cleaned docstring."""
    doc = inspect.getdoc(obj)
    return doc if doc else ""


def should_skip(name: str) -> bool:
    """Skip private/dunder members."""
    return name.startswith("_")


def get_public_members(module):
    """Get public classes and functions from module."""
    classes = []
    functions = []
    
    for name, obj in inspect.getmembers(module):
        if should_skip(name):
            continue
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            classes.append((name, obj))
        elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
            functions.append((name, obj))
    
    return classes, functions


def get_class_methods(cls) -> list:
    """Get public methods of a class."""
    methods = []
    for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction):
        if should_skip(name):
            continue
        methods.append((name, obj))
    return methods


def process_module(module_name: str) -> dict:
    """Process a module and extract documentation."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    
    classes, functions = get_public_members(module)
    
    return {
        "name": module_name,
        "doc": get_first_docstring_line(module),
        "classes": [
            {
                "name": name,
                "doc": get_first_docstring_line(cls),
                "full_doc": get_full_docstring(cls),
                "init_sig": get_signature(cls.__init__) if hasattr(cls, "__init__") else "()",
                "methods": [
                    {
                        "name": m_name,
                        "sig": get_signature(m_obj),
                        "doc": get_first_docstring_line(m_obj),
                    }
                    for m_name, m_obj in get_class_methods(cls)
                ],
            }
            for name, cls in classes
        ],
        "functions": [
            {
                "name": name,
                "sig": get_signature(func),
                "doc": get_first_docstring_line(func),
            }
            for name, func in functions
        ],
    }


def get_all_modules(package_name: str) -> list:
    """Recursively get all modules in a package."""
    modules = []
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return modules
    
    if not hasattr(package, "__path__"):
        return [package_name]
    
    modules.append(package_name)
    
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if name.split(".")[-1].startswith("_"):
            continue
        if is_pkg:
            modules.extend(get_all_modules(name))
        else:
            modules.append(name)
    
    return modules


def generate_markdown(modules_data: list) -> str:
    """Generate compact markdown documentation."""
    lines = [
        "# TorchEBM API Reference",
        "",
        "Complete API reference for LLM consumption.",
        "",
    ]
    
    for mod in modules_data:
        if not mod:
            continue
        
        # Module header
        short_name = mod["name"].replace("torchebm.", "")
        lines.append(f"## {short_name}")
        if mod["doc"]:
            lines.append(f"_{mod['doc']}_")
        lines.append("")
        
        # Classes
        for cls in mod["classes"]:
            lines.append(f"### `{cls['name']}`")
            if cls["doc"]:
                lines.append(f"{cls['doc']}")
            lines.append("")
            lines.append(f"```python")
            lines.append(f"{cls['name']}{cls['init_sig']}")
            lines.append(f"```")
            lines.append("")
            
            # Methods
            if cls["methods"]:
                lines.append("**Methods:**")
                for m in cls["methods"]:
                    doc_part = f" - {m['doc']}" if m["doc"] else ""
                    lines.append(f"- `{m['name']}{m['sig']}`{doc_part}")
                lines.append("")
        
        # Functions
        for func in mod["functions"]:
            lines.append(f"### `{func['name']}{func['sig']}`")
            if func["doc"]:
                lines.append(f"{func['doc']}")
            lines.append("")
    
    return "\n".join(lines)


def generate_context_file(modules_data: list) -> str:
    """Generate XML context file for LLM consumption."""
    lines = [
        '<torchebm title="TorchEBM API" summary="Energy-Based Modeling library for PyTorch">',
        "",
    ]
    
    for mod in modules_data:
        if not mod:
            continue
        
        short_name = mod["name"].replace("torchebm.", "")
        lines.append(f'<module name="{short_name}">')
        
        for cls in mod["classes"]:
            lines.append(f'  <class name="{cls["name"]}" init="{cls["init_sig"]}">')
            if cls["doc"]:
                lines.append(f'    <doc>{cls["doc"]}</doc>')
            for m in cls["methods"]:
                lines.append(f'    <method name="{m["name"]}" sig="{m["sig"]}"/>')
            lines.append(f'  </class>')
        
        for func in mod["functions"]:
            lines.append(f'  <function name="{func["name"]}" sig="{func["sig"]}"/>')
        
        lines.append(f'</module>')
        lines.append("")
    
    lines.append("</torchebm>")
    return "\n".join(lines)


def main():
    """Generate LLM documentation files."""
    print("Generating LLM-friendly documentation for TorchEBM...")
    
    # Get all modules
    all_modules = get_all_modules("torchebm")
    print(f"Found {len(all_modules)} modules")
    
    # Process modules
    modules_data = []
    for mod_name in sorted(all_modules):
        data = process_module(mod_name)
        if data and (data["classes"] or data["functions"]):
            modules_data.append(data)
            print(f"  Processed: {mod_name}")
    
    # Output paths
    output_dir = Path("docs/llm")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown
    md_content = generate_markdown(modules_data)
    md_path = output_dir / "all.md"
    md_path.write_text(md_content)
    print(f"\nGenerated: {md_path}")
    
    # Generate XML context
    ctx_content = generate_context_file(modules_data)
    ctx_path = output_dir / "llms-ctx.xml"
    ctx_path.write_text(ctx_content)
    print(f"Generated: {ctx_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
