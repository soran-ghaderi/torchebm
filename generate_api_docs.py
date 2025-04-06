#!/usr/bin/env python3
"""
Script to automatically generate API documentation files for TorchEBM.
This creates a structured set of markdown files for each module and submodule.
"""

import os
import re
import sys
import inspect
import importlib
import pkgutil
import shutil  # Add this import for directory operations


def is_package(module_name):
    """Check if a module is a package."""
    try:
        module = importlib.import_module(module_name)
        return hasattr(module, "__path__")
    except (ImportError, AttributeError):
        return False


def should_skip_member(name):
    """Determine if a member should be skipped in documentation."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def should_skip_module(name):
    """Determine if a module should be skipped in documentation."""
    return name.startswith("_")


def get_docstring_summary(obj):
    """Extract the first line of a docstring as a summary."""
    if not obj.__doc__:
        return "No description available."

    # Extract first line or first sentence
    doc = obj.__doc__.strip().split("\n")[0].strip()
    if len(doc) > 100:
        doc = doc[:97] + "..."
    return doc


def get_submodules(module_name):
    """Get all importable submodules of a module."""
    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, "__path__"):
            return []

        submodules = []
        for _, name, is_pkg in pkgutil.iter_modules(module.__path__, module_name + "."):
            # Skip modules starting with underscore
            if should_skip_module(name.split(".")[-1]):
                continue
            submodules.append((name, is_pkg))
        return submodules
    except ImportError:
        return []


def create_module_page(module_name, output_dir):
    """Create a markdown file for a module with class listings and summaries."""
    # Skip if module name starts with underscore
    if should_skip_module(module_name.split(".")[-1]):
        return None

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert module path to a directory path
    module_path = module_name.replace(".", "/")
    module_dir = os.path.join(output_dir, module_path)
    os.makedirs(module_dir, exist_ok=True)  # Create the actual module directory

    # Place the index.md file inside the module directory
    file_name = os.path.join(module_dir, "index.md")

    try:
        # Import the module
        module = importlib.import_module(module_name)
    except ImportError:
        import sys

        print("sys.version: ", sys.version)
        print(f"Warning: Could not import module {module_name}, skipping...")
        module = importlib.import_module(module_name)
        return None

    # Get module parts for breadcrumb
    module_parts = module_name.split(".")
    module_title = module_parts[-1].capitalize()

    # Create breadcrumb navigation
    if len(module_parts) > 1:
        breadcrumb = " > ".join([p.capitalize() for p in module_parts[:-1]])
        header = f"# {breadcrumb} > {module_title}\n\n"
    else:
        header = f"# {module_title}\n\n"

    # Get module docstring
    content = header  # Include the header with module title

    # Add table of contents section
    content += "## Contents\n\n"

    # Check if module is a package and get submodules/subpackages
    is_package = hasattr(module, "__path__")
    if is_package:
        submodules = get_submodules(module_name)
        subpackages = [name for name, is_pkg in submodules if is_pkg]
        modules = [name for name, is_pkg in submodules if not is_pkg]

        # Add subpackages to the table of contents
        if subpackages:
            content += "### Subpackages\n\n"
            for subpkg in sorted(subpackages):
                pkg_name = subpkg.split(".")[-1]
                content += f"- [{pkg_name.capitalize()}]({pkg_name}/)\n"
            content += "\n"

        # Add modules to the table of contents
        if modules:
            content += "### Modules\n\n"
            for mod in sorted(modules):
                mod_name = mod.split(".")[-1]
                summary = ""
                try:
                    mod_obj = importlib.import_module(mod)
                    if mod_obj.__doc__:
                        summary = " - " + get_docstring_summary(mod_obj)
                except ImportError:
                    pass
                content += f"- [{mod_name.capitalize()}]({mod_name}.md)\n"  # Changed to use actual module name
            content += "\n"

    # Find all classes in the module
    classes = []
    functions = []

    for name, obj in inspect.getmembers(module):
        if should_skip_member(name):
            continue

        if inspect.isclass(obj) and obj.__module__ == module_name:
            classes.append((name, obj))
        elif inspect.isfunction(obj) and obj.__module__ == module_name:
            functions.append((name, obj))

    # Add classes to the table of contents
    if classes:
        content += "### Classes\n\n"
        for name, cls in sorted(classes):
            summary = get_docstring_summary(cls)
            content += f"- [`{name}`](classes/{name}) - {summary}\n"

        content += "\n"

    # Add functions to the table of contents
    if functions:
        content += "### Functions\n\n"
        for name, func in sorted(functions):
            summary = get_docstring_summary(func)
            content += f"- [`{name}()`](#{name.lower()}) - {summary}\n"
        content += "\n"

    # Add the module API reference using mkdocstrings
    content += "## API Reference\n\n"
    content += f"::: {module_name}\n"
    content += "    options:\n"
    content += "      show_root_heading: true\n"
    content += "      show_root_toc_entry: true\n"
    content += "      show_source: true\n"

    content += "      show_symbol_type_heading: true\n"
    content += "      show_symbol_type_toc: true\n"
    content += "      show_docstring_attributes: false\n"
    content += "      show_docstring_classes: true\n"
    content += "      show_docstring_functions: true\n"
    content += "      trim_doctest_flags: true\n"
    content += "      show_category_heading: false\n"
    content += "      show_if_no_docstring: true\n"
    content += "      members_order: source\n"
    content += "      show_signature_annotations: true\n"
    content += "      separate_signature: true\n"
    content += "      unwrap_annotated: true\n"
    content += "      docstring_section_style: table\n"
    content += "      inherited_members: false\n"
    content += "      members:\n"
    content += '        - "!__*"\n'  # Exclude dunder methods

    # Write to file
    with open(file_name, "w") as f:
        f.write(content)

    print(f"Created documentation for {module_name} at {file_name}")

    # Generate individual class pages if there are classes
    class_pages = []
    for name, cls in classes:
        class_file = create_class_page(module_name, name, cls, output_dir)
        if class_file:
            class_pages.append((name, class_file))

    return os.path.relpath(file_name, "docs"), class_pages


def create_class_page(module_name, class_name, cls, output_dir):
    """Create a dedicated page for a class."""
    # Create directory for class pages
    module_path = module_name.replace(".", "/")
    class_dir = os.path.join(output_dir, module_path, "classes")
    os.makedirs(class_dir, exist_ok=True)

    file_name = os.path.join(class_dir, f"{class_name}.md")

    # Get module parts for breadcrumb
    module_parts = module_name.split(".")
    module_title = module_parts[-1].capitalize()

    # Create breadcrumb navigation
    if len(module_parts) > 1:
        breadcrumb = " > ".join([p.capitalize() for p in module_parts])
        header = f"# {breadcrumb} > {class_name}\n\n"
    else:
        header = f"# {module_title} > {class_name}\n\n"

    # Get class docstring
    class_doc = cls.__doc__ or "No description available."
    content = header + f"{class_doc.strip()}\n\n"
    content = ""

    # Add inheritance information
    if cls.__bases__ and cls.__bases__ != (object,):
        # base_classes = ", ".join(
        #     [b.__name__ for b in cls.__bases__ if b.__name__ != "object"]
        # )
        base_classes = []
        for base in cls.__bases__:
            if base.__name__ != "object":
                base_module = base.__module__
                # Fix the base class links to point to proper module pages
                if base_module.startswith(module_name.split(".")[0]):
                    # It's an internal link, create a proper relative path
                    base_module_path = base_module.replace(".", "/")
                    base_class_path = f"{base_module_path}.md"
                    # Check if it's likely a class in a module

                    if base_module != module_name:
                        base_classes.append(
                            f"[{base.__name__}](/{base_class_path}#{base_module.lower()}.{base.__name__.lower()})"
                        )
                    else:
                        # Same module, just use the class name
                        base_classes.append(
                            f"[{base.__name__}](#{base_module.lower()}.{base.__name__.lower()})"
                        )

                    # base_classes.append(
                    #     f"[{base.__name__}](/{base_module_path}#{base_module}.{base.__name__.lower()})"
                    # )
                else:
                    # External class, just show the name
                    base_classes.append(f"{base.__name__}")

        # if base_classes:
        # content += f"**Inherits from:** {base_classes}\n\n"
        # content += f"**Inherits from:** {', '.join(base_classes)}\n\n"

    # Add methods and attributes sections
    content += "## Methods and Attributes\n\n"
    content += f"::: {module_name}.{class_name}\n"
    content += "    options:\n"
    content += "      show_root_heading: false\n"
    content += "      show_source: true\n"
    content += "      members_order: source\n"
    content += "      show_category_heading: false\n"
    content += "      show_if_no_docstring: true\n"
    content += "      show_docstring_attributes: false\n"
    content += "      show_docstring_classes: true\n"
    content += "      show_docstring_functions: true\n"
    content += "      show_signature_annotations: true\n"
    content += "      separate_signature: true\n"
    content += "      unwrap_annotated: true\n"
    content += "      show_symbol_type_heading: true\n"
    content += "      show_symbol_type_toc: true\n"
    content += "      docstring_section_style: table\n"
    content += "      trim_doctest_flags: true\n"
    content += "      inherited_members: false\n"
    content += "      filters:\n"
    content += '        - "!^_[^_]"\n'  # Exclude single underscore methods
    content += '        - "!^__"\n'  # Exclude dunder methods

    # Write to file
    with open(file_name, "w") as f:
        f.write(content)

    print(f"Created documentation for class {class_name} at {file_name}")
    return os.path.relpath(file_name, "docs")


def build_hierarchical_structure(all_modules):
    """Build a hierarchical structure from modules and their classes."""
    hierarchy = {}

    # First pass: process all modules
    for module_info in all_modules:
        module_name = module_info["name"]

        if module_name.count(".") == 0:
            # This is the root package
            if module_name not in hierarchy:
                hierarchy[module_name] = {
                    "filepath": module_info["filepath"],
                    "title": module_name.capitalize(),
                    "subpackages": {},
                    "modules": {},
                    "classes": {},
                }
            # Add classes from the root module
            for class_name, class_path in module_info.get("classes", []):
                hierarchy[module_name]["classes"][class_name] = {
                    "filepath": class_path,
                    "title": class_name,
                }
            continue

        parts = module_name.split(".")
        root_package = parts[0]

        # Ensure the root package exists
        if root_package not in hierarchy:
            hierarchy[root_package] = {
                "filepath": "",
                "title": root_package.capitalize(),
                "subpackages": {},
                "modules": {},
                "classes": {},
            }

        current = hierarchy[root_package]

        # Navigate through the package hierarchy
        for i, part in enumerate(parts[1:], 1):
            current_path = ".".join(parts[: i + 1])

            # Check if we're at the last part (module name)
            if i == len(parts) - 1:
                if module_info["is_package"]:
                    if part not in current["subpackages"]:
                        current["subpackages"][part] = {
                            "filepath": module_info["filepath"],
                            "title": part.capitalize(),
                            "subpackages": {},
                            "modules": {},
                            "classes": {},
                        }
                    # Add classes to this package
                    for class_name, class_path in module_info.get("classes", []):
                        current["subpackages"][part]["classes"][class_name] = {
                            "filepath": class_path,
                            "title": class_name,
                        }
                else:
                    current["modules"][part] = {
                        "filepath": module_info["filepath"],
                        "title": part.capitalize(),
                        "classes": {},
                    }
                    # Add classes to this module
                    for class_name, class_path in module_info.get("classes", []):
                        current["modules"][part]["classes"][class_name] = {
                            "filepath": class_path,
                            "title": class_name,
                        }
            else:
                # This is a package level
                if part not in current["subpackages"]:
                    current["subpackages"][part] = {
                        "filepath": "",  # Will be filled if we have this package's info
                        "title": part.capitalize(),
                        "subpackages": {},
                        "modules": {},
                        "classes": {},
                    }
                current = current["subpackages"][part]

    return hierarchy


# def hierarchy_to_nav(hierarchy):
#     """Convert the hierarchical structure to MkDocs navigation format."""
#     nav = []
#
#     # Add overview first
#     nav.append({"Overview": "api/index.md"})
#
#     # Process each root package
#     for package_name, package_info in sorted(hierarchy.items()):
#         package_nav = package_to_nav(package_name, package_info)
#         nav.append(package_nav)
#
#     return nav


def hierarchy_to_nav(hierarchy):
    """Convert the hierarchical structure to MkDocs navigation format."""
    nav = []

    # Add overview first as a child item
    overview_item = {"Overview": "api/index.md"}

    # Process each root package
    packages = []
    for package_name, package_info in sorted(hierarchy.items()):
        if package_name == "torchebm":
            # Directly add subpackages and modules of "torchebm" to the navigation structure
            for subname, subinfo in sorted(package_info["subpackages"].items()):
                packages.append(package_to_nav(subname, subinfo))
            for modname, modinfo in sorted(package_info["modules"].items()):
                packages.append(package_to_nav(modname, modinfo))
        else:
            package_nav = package_to_nav(package_name, package_info)
            packages.append(package_nav)

    # Include overview as the first child item, followed by packages
    all_items = [overview_item] + packages

    return all_items


def package_to_nav(name, info):
    """Convert a package structure to navigation format."""
    # If this is a simple entry with no subpackages, modules, or classes
    if not info["subpackages"] and not info["modules"] and not info["classes"]:
        return {info["title"]: info["filepath"]}

    # Otherwise, it's a section with sub-entries
    section = {info["title"]: []}

    # Add subpackages (recursive)
    for subname, subinfo in sorted(info["subpackages"].items()):
        section[info["title"]].append(package_to_nav(subname, subinfo))

    # Add modules
    for modname, modinfo in sorted(info["modules"].items()):
        # If the module has classes, create a subsection
        if modinfo["classes"]:
            module_section = {modinfo["title"]: [modinfo["filepath"]]}
            # Add classes to the module section
            for classname, classinfo in sorted(modinfo["classes"].items()):
                module_section[modinfo["title"]].append(
                    {classinfo["title"]: classinfo["filepath"]}
                )
            section[info["title"]].append(module_section)
        else:
            # Simple module with no classes
            section[info["title"]].append({modinfo["title"]: modinfo["filepath"]})

        # section[info["title"]].append({modinfo["title"]: modinfo["filepath"]})
        #
        # # Add direct classes of the package (if any)
        # for classname, classinfo in sorted(info["classes"].items()):
        #     section[info["title"]].append({classinfo["title"]: classinfo["filepath"]})
    return section

    return section


def generate_index_page(hierarchy, output_dir):
    """Generate the index page with a hierarchical overview."""
    content = "# API Reference\n\n"
    content += "This page provides an overview of the TorchEBM API.\n\n"

    # Create a table of contents
    content += "## Table of Contents\n\n"

    # Add the main packages to the TOC
    for package_name, package_info in sorted(hierarchy.items()):
        # content += f"- [{package_info['title']}](#{package_name.lower()})\n"

        #  =======================================s
        content += f"## {package_info['title']} <a id='{package_name.lower()}'></a>\n\n"

        if package_info["filepath"]:
            content += f"[Package Documentation]({os.path.relpath(package_info['filepath'], 'docs')})\n\n"

        #  =======================================e

        # Add subpackages
        if package_info["subpackages"]:
            for subname, subinfo in sorted(package_info["subpackages"].items()):
                content += f"  - [{subinfo['title']}](#{package_name.lower()}-{subname.lower()})\n"

        # Add modules (only direct modules of the package)
        if package_info["modules"]:
            for modname, modinfo in sorted(package_info["modules"].items()):
                content += f"  - [{modinfo['title']}](#{package_name.lower()}-{modname.lower()})\n"

    content += "\n"

    # Generate detailed content for each package
    for package_name, package_info in sorted(hierarchy.items()):
        content += f"## {package_info['title']} <a id='{package_name.lower()}'></a>\n\n"

        if package_info["filepath"]:
            content += f"[Package Documentation]({os.path.relpath(package_info['filepath'], 'docs')})\n\n"

        # Add description if we can get it
        try:
            package = importlib.import_module(package_name)
            if package.__doc__:
                content += f"{package.__doc__.strip().split('.')[0]}.\n\n"
        except (ImportError, AttributeError, IndexError):
            pass

        # Add subpackages
        if package_info["subpackages"]:
            content += "### Subpackages\n\n"
            for subname, subinfo in sorted(package_info["subpackages"].items()):
                anchor = f"{package_name.lower()}-{subname.lower()}"
                content += f"#### {subinfo['title']} <a id='{anchor}'></a>\n\n"
                if subinfo["filepath"]:
                    content += f"[Documentation]({os.path.relpath(subinfo['filepath'], 'docs')})\n\n"

                # Add modules in this subpackage
                if subinfo["modules"]:
                    content += "**Modules:**\n\n"
                    for modname, modinfo in sorted(subinfo["modules"].items()):
                        content += f"- [{modinfo['title']}]({os.path.relpath(modinfo['filepath'], 'docs')}) - "

                        # Try to get a description
                        try:
                            mod = importlib.import_module(
                                f"{package_name}.{subname}.{modname}"
                            )
                            if mod.__doc__:
                                content += f"{mod.__doc__.strip().split('.')[0]}.\n"
                            else:
                                content += "No description available.\n"
                        except (ImportError, AttributeError, IndexError):
                            content += "No description available.\n"

                    content += "\n"

                # Add direct classes in this subpackage
                if subinfo["classes"]:
                    content += "**Classes:**\n\n"
                    for classname, classinfo in sorted(subinfo["classes"].items()):
                        content += f"- [{classinfo['title']}]({os.path.relpath(classinfo['filepath'], 'docs')})\n"

                    content += "\n"

        # Add modules directly in the package
        if package_info["modules"]:
            content += "### Modules\n\n"
            for modname, modinfo in sorted(package_info["modules"].items()):
                anchor = f"{package_name.lower()}-{modname.lower()}"
                content += f"#### {modinfo['title']} <a id='{anchor}'></a>\n\n"
                content += f"[Documentation]({os.path.relpath(modinfo['filepath'], 'docs')})\n\n"

                # Add classes in this module
                if modinfo["classes"]:
                    content += "**Classes:**\n\n"
                    for classname, classinfo in sorted(modinfo["classes"].items()):
                        content += f"- [{classinfo['title']}]({os.path.relpath(classinfo['filepath'], 'docs')})\n"

                    content += "\n"

        # Add direct classes in this package
        if package_info["classes"]:
            content += "### Classes\n\n"
            for classname, classinfo in sorted(package_info["classes"].items()):
                content += f"- [{classinfo['title']}]({os.path.relpath(classinfo['filepath'], 'docs')})\n"

            content += "\n"

    # Write to file
    with open(os.path.join(output_dir, "index.md"), "w") as f:
        f.write(content)

    print(f"Created API documentation index at {os.path.join(output_dir, 'index.md')}")


def update_mkdocs_nav(nav_structure, tab_name="API Reference"):
    """Update the mkdocs.yml file with the new navigation structure."""
    try:
        # Read existing mkdocs.yml as text
        with open("mkdocs.yml", "r") as f:
            lines = f.readlines()

        # Find the nav section and API subsection
        nav_section_start = -1
        api_section_start = -1
        api_section_end = -1
        in_nav = False
        nav_indent = -1

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("nav:"):
                nav_section_start = i
                in_nav = True
                nav_indent = len(line) - len(stripped)
                continue

            if in_nav:
                if stripped and not stripped.startswith("#"):  # Skip comments
                    current_indent = len(line) - len(stripped)

                    # Back to same level as nav or less, we've exited nav section
                    if current_indent <= nav_indent and i > nav_section_start + 1:
                        in_nav = False
                        continue

                    # Look for API section
                    if stripped.startswith(f"- {tab_name}:"):
                        api_section_start = i
                        # Find where API section ends
                        for j in range(i + 1, len(lines)):
                            if j >= len(lines):
                                api_section_end = len(lines)
                                break

                            next_line = lines[j].lstrip()
                            if next_line and not next_line.startswith("#"):
                                next_indent = len(lines[j]) - len(next_line)
                                # If at same level as "- API:" or less, we've exited the API section
                                if next_indent <= current_indent:
                                    api_section_end = j
                                    break

                        if api_section_end == -1:  # API is last section
                            api_section_end = len(lines)
                        break

        # Convert nav_structure to YAML lines with proper indentation
        api_yaml = yaml.safe_dump(nav_structure, default_flow_style=False).split("\n")
        base_indent = "  "  # Base indentation for nav items

        if api_section_start != -1:
            # Get the indent of the "- API:" line
            api_line_indent = len(lines[api_section_start]) - len(
                lines[api_section_start].lstrip()
            )
            base_indent = " " * (api_line_indent + 2)  # Add 2 spaces for sub-items

        api_indented_yaml = []
        for line in api_yaml:
            if line.strip():
                api_indented_yaml.append(base_indent + line + "\n")

        # Update the file
        if api_section_start != -1:
            # Replace existing API section
            new_lines = lines[: api_section_start + 1]  # Include the "- API:" line
            new_lines.extend(api_indented_yaml)
            new_lines.extend(lines[api_section_end:])
        elif nav_section_start != -1:
            # Add new API section to existing nav
            api_entry = " " * (nav_indent + 2) + f"- {tab_name}:\n"

            # Find where to insert (end of nav)
            insert_pos = nav_section_start + 1
            while insert_pos < len(lines) and (
                not lines[insert_pos].strip()
                or lines[insert_pos].strip().startswith("#")
                or len(lines[insert_pos]) - len(lines[insert_pos].lstrip()) > nav_indent
            ):
                insert_pos += 1

            new_lines = lines[:insert_pos]
            new_lines.append(api_entry)
            new_lines.extend(api_indented_yaml)
            new_lines.extend(lines[insert_pos:])
        else:
            # Create new nav section
            nav_entry = "nav:\n"
            api_entry = f"  - {tab_name}:\n"

            new_lines = lines
            new_lines.append(nav_entry)
            new_lines.append(api_entry)
            new_lines.extend(api_indented_yaml)

        # Write updated content back
        with open("mkdocs.yml", "w") as f:
            f.writelines(new_lines)

        print("Successfully updated mkdocs.yml with new API navigation structure")

    except Exception as e:
        print(f"Error updating mkdocs.yml: {e}")
        print("Please manually update your navigation structure")

        # Create backup file with nav structure
        backup_file = "api_nav_structure.yml"
        with open(backup_file, "w") as f:
            yaml.dump({f"{tab_name}": nav_structure}, f, default_flow_style=False)
        print(f"Navigation structure saved to {backup_file}")


# Add a new function to ensure proper inline URLs in the documentation:
def add_inline_class_references(module_name, output_dir):
    """Add proper inline references to classes in module documentation."""
    module_path = module_name.replace(".", "/")
    actual_module_name = module_name.split(".")[-1]

    # Fix: Correctly build the path to the module documentation file
    module_dir_path = os.path.dirname(os.path.join(output_dir, module_path))
    file_name = os.path.join(module_dir_path, f"{actual_module_name}.md")

    if not os.path.exists(file_name):
        return

    with open(file_name, "r") as f:
        content = f.read()

    # Find class reference patterns and add fully qualified IDs
    pattern = r"## API Reference\n\n::: " + module_name
    if re.search(pattern, content):
        # Add anchors with fully qualified names for each class
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and obj.__module__ == module_name
                    and not should_skip_member(name)
                ):
                    qualified_anchor = f"{module_name}.{name}"
                    anchor_line = f'<a id="{qualified_anchor}"></a>\n'
                    simple_anchor_line = f'<a id="{name}"></a>\n'
                    # Add both anchors to support both types of links
                    content = re.sub(
                        f"### {name}\n",
                        f"### {name}\n{anchor_line}{simple_anchor_line}",
                        content,
                    )

        except ImportError:
            pass

    with open(file_name, "w") as f:
        f.write(content)


def process_package(package_name, output_dir):
    """Process a package and all its submodules, generating documentation."""
    all_modules = []

    try:
        # Import the root package
        package = importlib.import_module(package_name)
    except ImportError:
        print(f"Error: Could not import package {package_name}")
        return []

    # Create documentation for the root package
    root_filepath, root_classes = create_module_page(package_name, output_dir)
    if root_filepath:
        all_modules.append(
            {
                "name": package_name,
                "filepath": root_filepath,
                "is_package": True,
                "classes": root_classes,
            }
        )

    # Queue for BFS traversal of packages
    queue = [(package_name, True)]
    visited = {package_name}

    # Process all packages and modules using BFS
    while queue:
        current_name, is_pkg = queue.pop(0)

        if is_pkg:
            # Get all importable submodules
            submodules = get_submodules(current_name)

            for submodule_name, is_subpkg in submodules:
                if submodule_name not in visited:
                    visited.add(submodule_name)
                    queue.append((submodule_name, is_subpkg))

                    # Create documentation for this submodule
                    print(
                        "creat module page for submodule_name, output_dir",
                        submodule_name,
                        output_dir,
                    )
                    filepath, classes = create_module_page(submodule_name, output_dir)
                    if filepath:
                        all_modules.append(
                            {
                                "name": submodule_name,
                                "filepath": filepath,
                                "is_package": is_subpkg,
                                "classes": classes,
                            }
                        )
    # After creating all documentation, add inline references
    for module_info in all_modules:
        add_inline_class_references(module_info["name"], output_dir)

    return all_modules


def clean_api_directory(output_dir, preserve_files=None):
    """Clean up the API directory while preserving specified files.

    Args:
        output_dir (str): Path to the API documentation directory
        preserve_files (list): List of filenames to preserve (relative to output_dir)
    """
    if preserve_files is None:
        preserve_files = []

    # Ensure output_dir exists
    if not os.path.exists(output_dir):
        return

    # Convert preserve_files to absolute paths
    preserve_paths = {os.path.join(output_dir, f) for f in preserve_files}

    # Walk through directory and remove files/dirs
    for root, dirs, files in os.walk(output_dir, topdown=False):
        # Remove files first
        for name in files:
            filepath = os.path.join(root, name)
            if filepath not in preserve_paths:
                os.remove(filepath)

        # Then remove empty directories
        for name in dirs:
            dirpath = os.path.join(root, name)
            try:
                os.rmdir(dirpath)  # This will only remove empty directories
            except OSError:
                # Directory not empty (probably contains preserved files)
                pass


def generate_api_docs(package_name, output_dir="docs/api"):
    """Generate documentation for a package and its submodules with hierarchical structure."""
    print(f"Generating API documentation for {package_name}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Clean up existing files while preserving index.md and package root doc if they exist
    preserve_files = []
    if os.path.exists(os.path.join(output_dir, "index.md")):
        preserve_files.append("index.md")
    package_doc = f"{package_name}.md"
    if os.path.exists(os.path.join(output_dir, package_doc)):
        preserve_files.append(package_doc)

    clean_api_directory(output_dir, preserve_files)

    # Process the entire package
    all_modules = process_package(package_name, output_dir)

    if not all_modules:
        print(
            "No modules were processed. Please check the package name and permissions."
        )
        return

    print(f"Processed {len(all_modules)} modules and packages.")

    # Build hierarchical structure
    hierarchy = build_hierarchical_structure(all_modules)

    # Generate the index page only if it doesn't exist
    if "index.md" not in preserve_files:
        generate_index_page(hierarchy, output_dir)
    else:
        print(f"Preserved existing index.md in {output_dir}")

    # Convert hierarchy to navigation structure
    nav_structure = hierarchy_to_nav(hierarchy)

    # Update mkdocs.yml
    update_mkdocs_nav(nav_structure, tab_name="API Reference")

    print("\nAPI documentation generation complete!")
    print(f"- Generated documentation for {len(all_modules)} modules")
    print(f"- Created class documentation pages for all public classes")
    print(f"- Updated navigation structure in mkdocs.yml")
    if "index.md" in preserve_files:
        print(f"- Preserved existing index at {os.path.join(output_dir, 'index.md')}")
    else:
        print(f"- Created hierarchical index at {os.path.join(output_dir, 'index.md')}")


if __name__ == "__main__":
    # Check for PyYAML
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required. Please install it with:")
        print("pip install pyyaml")
        sys.exit(1)

    # Generate API docs for torchebm
    generate_api_docs("torchebm")
