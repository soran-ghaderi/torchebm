#!/usr/bin/env python3
"""
Script to automatically generate API documentation files for TorchEBM.
This creates a structured set of markdown files for each module and submodule.
"""

import os
import importlib
import inspect
import pkgutil
import yaml
import sys
from collections import defaultdict


def is_package(module_name):
    """Check if a module is a package."""
    try:
        module = importlib.import_module(module_name)
        return hasattr(module, "__path__")
    except (ImportError, AttributeError):
        return False


def create_module_page(module_name, output_dir, nav_items=None):
    """Create a markdown file for a module."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert module path to a file path
    module_path = module_name.replace(".", "/")
    file_name = os.path.join(output_dir, f"{module_path}.md")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Create markdown content
    # module_title = module_name.split(".")[-1].capitalize()
    module_parts = module_name.split(".")
    module_title = module_parts[-1].capitalize()

    if len(module_parts) > 1:
        breadcrumb = " > ".join([p.capitalize() for p in module_parts[:-1]])
        content = f"# {breadcrumb} > {module_title}\n\n"
    else:
        content = f"# {module_title}\n\n"

    content = f"# {module_title} Module\n\n"
    content += f"::: {module_name}\n"
    content += "    options:\n"
    content += "      show_root_heading: true\n"
    content += "      show_root_toc_entry: true\n"
    content += "      show_submodules: true\n"

    # Write to file
    with open(file_name, "w") as f:
        f.write(content)

    print(f"Created documentation for {module_name} at {file_name}")

    # # Add to nav items if provided
    # if nav_items is not None:
    #     relative_path = os.path.relpath(file_name, "docs")
    #     nav_items.append(f"    - {module_title}: {relative_path}")

    return os.path.relpath(file_name, "docs")


# def generate_api_docs(package_name, output_dir="docs/api"):
#     """Generate markdown files for a package and its submodules."""
#     # Create main index file
#     os.makedirs(output_dir, exist_ok=True)
#
#     nav_items = []
#
#     # Create index.md file
#     index_content = f"# {package_name.capitalize()} API Reference\n\n"
#     index_content += "This page provides an overview of the API.\n\n"
#
#     # Try to import the package
#     try:
#         package = importlib.import_module(package_name)
#     except ImportError:
#         print(f"Error: Could not import package {package_name}")
#         return
#
#     # Process the package and its submodules
#     for _, module_name, is_pkg in pkgutil.walk_packages(
#         package.__path__, package.__name__ + "."
#     ):
#         # Create a page for this module
#         create_module_page(module_name, output_dir, nav_items)
#
#         # If it's a package, we've already created pages for its submodules
#         # through the walk_packages function
#
#     # Create a root page for the main package
#     create_module_page(package_name, output_dir, nav_items)
#
#     # Update index with module listing
#     index_content += "## Modules\n\n"
#     for nav_item in sorted(nav_items):
#         print("here is the nav item: ", nav_item)
#         module_name = nav_item.split(":")[0].split("-")[1].strip()
#         index_content += f"- [{module_name}]({nav_item.split(':')[1].strip()})\n"
#
#     # Write index file
#     with open(os.path.join(output_dir, "index.md"), "w") as f:
#         f.write(index_content)
#
#     print(f"Created API documentation index at {os.path.join(output_dir, 'index.md')}")
#
#     # Print nav configuration
#     print("\nAdd this to your mkdocs.yml nav section:")
#     print("  - API:")
#     print(f"    - Overview: api/index.md")
#     for item in sorted(nav_items):
#         print(item)
#
#
# if __name__ == "__main__":
#     # Generate API docs for torchebm
#     generate_api_docs("torchebm")


# def update_mkdocs_nav(nav_structure):
#     """Update the mkdocs.yml file with the new navigation structure."""
#     try:
#         # Read existing mkdocs.yml
#         with open("mkdocs.yml", "r") as f:
#             mkdocs_config = yaml.safe_load(f)
#
#         # Find the API section in nav
#         api_section_found = False
#         for i, section in enumerate(mkdocs_config.get("nav", [])):
#             if isinstance(section, dict) and "API" in section:
#                 mkdocs_config["nav"][i]["API"] = nav_structure
#                 api_section_found = True
#                 break
#
#         # If no API section found, add one
#         if not api_section_found:
#             api_entry = {"API": nav_structure}
#             if "nav" not in mkdocs_config:
#                 mkdocs_config["nav"] = []
#             mkdocs_config["nav"].append(api_entry)
#
#         # Write updated config back
#         with open("mkdocs.yml", "w") as f:
#             yaml.dump(mkdocs_config, f, default_flow_style=False, sort_keys=False)
#
#         print("Updated mkdocs.yml with new API navigation structure")
#     except Exception as e:
#         print(f"Error updating mkdocs.yml: {e}")
#         print("Please manually update your navigation structure.")


# def update_mkdocs_nav(nav_structure):
#     """Update the mkdocs.yml file with the new navigation structure."""
#     try:
#         # Read existing mkdocs.yml preserving comments and formatting
#         with open("mkdocs.yml", "r") as f:
#             mkdocs_content = f.read()
#
#         # Load YAML content
#         mkdocs_config = yaml.safe_load(mkdocs_content)
#
#         # Find the API section in nav
#         api_section_found = False
#         for i, section in enumerate(mkdocs_config.get("nav", [])):
#             if isinstance(section, dict) and "API" in section:
#                 # Replace only the API section
#                 mkdocs_config["nav"][i]["API"] = nav_structure
#                 api_section_found = True
#                 break
#
#         print("mkdocs_config", mkdocs_config)
#         print("nav_structure", nav_structure)
#         # If no API section found, add one
#         if not api_section_found:
#             api_entry = {"API": nav_structure}
#             if "nav" not in mkdocs_config:
#                 mkdocs_config["nav"] = []
#             mkdocs_config["nav"].append(api_entry)
#
#         # Write updated config back with proper formatting
#         with open("mkdocs.yml", "w") as f:
#             yaml.dump(
#                 mkdocs_config,
#                 f,
#                 default_flow_style=False,
#                 sort_keys=False,
#                 indent=2,
#                 width=80,
#                 allow_unicode=True,
#             )
#
#         print("Successfully updated mkdocs.yml with new API navigation structure")
#
#         # Output the new structure for verification
#         print("\nNew API navigation structure:")
#         for entry in nav_structure:
#             if isinstance(entry, dict):
#                 for k, v in entry.items():
#                     print(f"- {k}")
#                     if isinstance(v, list):
#                         for item in v:
#                             if isinstance(item, dict):
#                                 for sk, sv in item.items():
#                                     print(f"  - {sk}")
#                             else:
#                                 print(f"  - {item}")
#                     else:
#                         print(f"  {v}")
#             else:
#                 print(f"- {entry}")
#
#     except Exception as e:
#         print(f"Error updating mkdocs.yml: {e}")
#         print("Please manually update your navigation structure using the following:")
#         print("\nAPI navigation structure to add manually:")
#         print(yaml.dump({"API": nav_structure}, default_flow_style=False))
#
#         # Create a backup file with the nav structure
#         backup_file = "api_nav_structure.yml"
#         with open(backup_file, "w") as f:
#             yaml.dump({"API": nav_structure}, f, default_flow_style=False)
#         print(f"Navigation structure saved to {backup_file}")


import re
import yaml


def update_mkdocs_nav(nav_structure):
    """Update the mkdocs.yml file with the new navigation structure using text processing."""
    try:
        # Read existing mkdocs.yml as text
        with open("mkdocs.yml", "r") as f:
            lines = f.readlines()

        # Find the API section in the nav
        api_section_start = -1
        api_section_end = -1
        nav_section_start = -1
        nav_section_indent = -1
        in_nav = False
        current_indent = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("nav:"):
                nav_section_start = i
                in_nav = True
                nav_section_indent = len(line) - len(stripped)
                continue

            if in_nav:
                # Calculate the current indent level
                if stripped and not stripped.startswith("#"):  # Skip comments
                    current_indent = len(line) - len(stripped)

                    # If we're back to the same indent level as nav or less, we've exited the nav section
                    if (
                        current_indent <= nav_section_indent
                        and i > nav_section_start + 1
                    ):
                        in_nav = False
                        continue

                    # Look for the API section
                    if stripped.startswith("- API:"):
                        api_section_start = i
                        # Find where the API section ends
                        for j in range(i + 1, len(lines)):
                            next_line = lines[j].lstrip()
                            if next_line and not next_line.startswith(
                                "#"
                            ):  # Skip comments
                                next_indent = len(lines[j]) - len(next_line)
                                # If indent level is the same as or less than the "- API:" line,
                                # we've reached the end of the API section
                                if next_indent <= current_indent:
                                    api_section_end = j
                                    break
                        if (
                            api_section_end == -1
                        ):  # If API section is the last in the file
                            api_section_end = len(lines)
                        break

        # Convert nav_structure to YAML format
        api_yaml = yaml.safe_dump(nav_structure, default_flow_style=False).split("\n")
        # Properly indent the YAML lines
        api_indented_yaml = []
        base_indent = " " * (current_indent + 2)  # Add 2 spaces for the subsection
        for line in api_yaml:
            if line.strip():
                api_indented_yaml.append(base_indent + line + "\n")

        # If we found the API section, replace it
        if api_section_start != -1:
            # Replace the API section with the new content
            new_lines = lines[: api_section_start + 1]  # Include the "- API:" line
            new_lines.extend(api_indented_yaml)
            new_lines.extend(lines[api_section_end:])

        else:
            # If we didn't find an API section but found the nav section, add it
            if nav_section_start != -1:
                api_entry = (
                    " " * (nav_section_indent + 2) + "- API:\n"
                )  # Proper indent for nav subsection

                # Find where to insert the API section (end of nav)
                insert_pos = nav_section_start + 1
                while insert_pos < len(lines) and (
                    not lines[insert_pos].strip()
                    or lines[insert_pos].strip().startswith("#")
                    or len(lines[insert_pos]) - len(lines[insert_pos].lstrip())
                    > nav_section_indent
                ):
                    insert_pos += 1

                new_lines = lines[:insert_pos]
                new_lines.append(api_entry)
                new_lines.extend(api_indented_yaml)
                new_lines.extend(lines[insert_pos:])
            else:
                # If there's no nav section at all, add it at the end
                nav_entry = "nav:\n"
                api_entry = "  - API:\n"

                new_lines = lines
                new_lines.append(nav_entry)
                new_lines.append(api_entry)
                new_lines.extend(api_indented_yaml)

        # Write the updated content back to the file
        with open("mkdocs.yml", "w") as f:
            f.writelines(new_lines)

        print("Successfully updated mkdocs.yml with new API navigation structure")

    except Exception as e:
        print(f"Error updating mkdocs.yml: {e}")
        print("Please manually update your navigation structure using the following:")
        print("\nAPI navigation structure to add manually:")
        print(yaml.dump({"API": nav_structure}, default_flow_style=False))

        # Create a backup file with the nav structure
        backup_file = "api_nav_structure.yml"
        with open(backup_file, "w") as f:
            yaml.dump({"API": nav_structure}, f, default_flow_style=False)
        print(f"Navigation structure saved to {backup_file}")


def build_hierarchical_structure(all_modules):
    """Build a hierarchical structure from a flat list of modules."""
    hierarchy = {}

    # First pass: identify all packages and create the hierarchy
    for module_name, file_path in all_modules.items():
        if module_name.count(".") == 0:
            # This is the root package
            if module_name not in hierarchy:
                hierarchy[module_name] = {
                    "filepath": file_path,
                    "title": module_name.capitalize(),
                    "subpackages": {},
                    "modules": {},
                }
            continue

        parts = module_name.split(".")
        current_package = parts[0]

        # Ensure the root package exists
        if current_package not in hierarchy:
            hierarchy[current_package] = {
                "filepath": all_modules.get(current_package, ""),
                "title": current_package.capitalize(),
                "subpackages": {},
                "modules": {},
            }

        current = hierarchy[current_package]

        # Navigate through the package hierarchy
        for i, part in enumerate(parts[1:], 1):
            # Check if we're at the last part (the module name)
            if i == len(parts) - 1:
                # This is a module, add it to the current level
                current_path = ".".join(parts[: i + 1])
                if is_package(current_path):
                    # It's a subpackage
                    if part not in current["subpackages"]:
                        current["subpackages"][part] = {
                            "filepath": file_path,
                            "title": part.capitalize(),
                            "subpackages": {},
                            "modules": {},
                        }
                else:
                    # It's a module
                    current["modules"][part] = {
                        "filepath": file_path,
                        "title": part.capitalize(),
                    }
            else:
                # This is a package level, ensure it exists
                if part not in current["subpackages"]:
                    current_path = ".".join(parts[: i + 1])
                    current["subpackages"][part] = {
                        "filepath": all_modules.get(current_path, ""),
                        "title": part.capitalize(),
                        "subpackages": {},
                        "modules": {},
                    }
                current = current["subpackages"][part]

    return hierarchy


def hierarchy_to_nav(hierarchy):
    """Convert the hierarchical structure to MkDocs navigation format."""
    nav = []

    # Add overview first
    nav.append({"Overview": "api/index.md"})

    # Process each root package
    for package_name, package_info in sorted(hierarchy.items()):
        package_nav = package_to_nav(package_name, package_info)
        nav.append(package_nav)

    return nav


def package_to_nav(name, info):
    """Convert a package structure to navigation format."""
    # If this package has no subpackages or modules, it's a simple entry
    if not info["subpackages"] and not info["modules"]:
        return {info["title"]: info["filepath"]}

    # Otherwise, it's a section with sub-entries
    section = {info["title"]: []}

    # Add the package itself as an entry if it has a filepath
    if info["filepath"]:
        section[info["title"]].append({"Package Overview": info["filepath"]})

    # Add subpackages (recursive)
    for subname, subinfo in sorted(info["subpackages"].items()):
        section[info["title"]].append(package_to_nav(subname, subinfo))

    # Add modules
    for modname, modinfo in sorted(info["modules"].items()):
        section[info["title"]].append({modinfo["title"]: modinfo["filepath"]})

    return section


def generate_index_page(hierarchy, output_dir):
    """Generate the index page with a hierarchical overview."""
    content = "# API Reference\n\n"
    content += "This page provides an overview of the TorchEBM API.\n\n"

    for package_name, package_info in sorted(hierarchy.items()):
        content += f"## {package_info['title']}\n\n"

        if package_info["filepath"]:
            content += f"[Package Overview]({os.path.relpath(package_info['filepath'], 'docs')})\n\n"

        if package_info["subpackages"] or package_info["modules"]:
            content += "### Contents\n\n"

            # Add subpackages
            if package_info["subpackages"]:
                content += "#### Subpackages\n\n"
                for subname, subinfo in sorted(package_info["subpackages"].items()):
                    content += f"- [{subinfo['title']}]({os.path.relpath(subinfo['filepath'], 'docs')})\n"
                content += "\n"

            # Add modules
            if package_info["modules"]:
                content += "#### Modules\n\n"
                for modname, modinfo in sorted(package_info["modules"].items()):
                    content += f"- [{modinfo['title']}]({os.path.relpath(modinfo['filepath'], 'docs')})\n"
                content += "\n"

    # Write to file
    with open(os.path.join(output_dir, "index.md"), "w") as f:
        f.write(content)

    print(f"Created API documentation index at {os.path.join(output_dir, 'index.md')}")


def generate_api_docs(package_name, output_dir="docs/api"):
    """Generate markdown files for a package and its submodules with hierarchical structure."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store all modules and their file paths
    all_modules = {}

    # Try to import the package
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        print(f"Error: Could not import package {package_name}")
        return

    # Create a page for the root package
    root_filepath = create_module_page(package_name, output_dir)
    all_modules[package_name] = root_filepath

    # Process the package and its submodules
    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        # Create a page for this module
        filepath = create_module_page(module_name, output_dir)
        all_modules[module_name] = filepath

    # Build hierarchical structure
    hierarchy = build_hierarchical_structure(all_modules)

    # Generate the index page
    generate_index_page(hierarchy, output_dir)

    # Convert hierarchy to navigation structure
    nav_structure = hierarchy_to_nav(hierarchy)

    # Update mkdocs.yml
    update_mkdocs_nav(nav_structure)

    print("\nAPI documentation generation complete!")
    print(f"- Generated documentation for {len(all_modules)} modules")
    print(f"- Updated navigation structure in mkdocs.yml")
    print(f"- Created hierarchical index at {os.path.join(output_dir, 'index.md')}")


if __name__ == "__main__":
    # Check if PyYAML is installed
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required. Please install it with:")
        print("pip install pyyaml")
        sys.exit(1)

    # Generate API docs for torchebm
    generate_api_docs("torchebm")
