# Developer Guide for `torchebm`

## Table of Contents

1. [Introduction](#introduction)
2. [Setting Up the Development Environment](#setting-up-the-development-environment)
   - [Prerequisites](#prerequisites)
   - [Cloning the Repository](#cloning-the-repository)
   - [Installing Dependencies](#installing-dependencies)
3. [Code Style and Quality](#code-style-and-quality)
   - [Code Formatting](#code-formatting)
   - [Linting](#linting)
   - [Type Checking](#type-checking)
   - [Testing](#testing)
4. [Documentation](#documentation)
   - [Docstring Conventions](#docstring-conventions)
   - [API Documentation Generation](#api-documentation-generation)
5. [Contributing](#contributing)
   - [Branching Strategy](#branching-strategy)
   - [Commit Messages](#commit-messages)
   - [Pull Requests](#pull-requests)
6. [Additional Resources](#additional-resources)

## Introduction

Welcome to the developer guide for `torchebm`, a Python library focused on components and algorithms for energy-based models. This document provides instructions and best practices for contributing to the project.

## Setting Up the Development Environment

### Prerequisites

Ensure you have the following installed:

- **Python**: Version 3.9 or higher.
- **Git**: For version control.
- **pip**: Python package installer.

### Cloning the Repository

Clone the repository using SSH:

```bash
git clone git@github.com:soran-ghaderi/torchebm.git
```

Or using HTTPS:

```bash
git clone https://github.com/soran-ghaderi/torchebm.git
```

Navigate to the project directory:

```bash
cd torchebm
```

### Installing Dependencies

Install the development dependencies:

```bash
pip install -e .[dev]
```

This command installs the package in editable mode along with all development dependencies specified in the `pyproject.toml` file.

## Code Style and Quality

Maintaining a consistent code style ensures readability and ease of collaboration.

To streamline code formatting and quality checks in your development workflow, integrating tools like **Black**, **isort**, and **mypy** directly into your Integrated Development Environment (IDE) can be highly beneficial. Many modern IDEs, such as **PyCharm**, offer built-in support or plugins for these tools, enabling automatic code formatting and linting as you write.

**PyCharm Integration:**

- **Black Integration:** Starting from PyCharm 2023.2, Black integration is built-in. To enable it:

  1. Navigate to `Preferences` or `Settings` > `Tools` > `Black`.
  2. Configure the settings as desired.
  3. Ensure Black is installed in your environment:

     ```bash
     pip install black
     ```

- **isort Integration:** While PyCharm doesn't have built-in isort support, you can set it up using File Watchers:

  1. Install the File Watchers plugin if it's not already installed.
  2. Navigate to `Preferences` or `Settings` > `Tools` > `File Watchers`.
  3. Add a new watcher for isort with the following configuration:
     - **File Type:** Python
     - **Scope:** Current File
     - **Program:** Path to isort executable (e.g., `$PyInterpreterDirectory$/isort`)
     - **Arguments:** `$FilePath$`
     - **Output Paths to Refresh:** `$FilePath$`
     - **Working Directory:** `$ProjectFileDir$`

- **mypy Integration:** To integrate mypy:

  1. Install mypy in your environment:

     ```bash
     pip install mypy
     ```

  2. Set up a File Watcher in PyCharm similar to isort, replacing the program path with the mypy executable path.

**VSCode Integration:**

For Visual Studio Code users, extensions are available for seamless integration:

- **Black Formatter:** Install the "Python" extension by Microsoft, which supports Black formatting.
- **isort:** Use the "Python" extension's settings to enable isort integration.
- **mypy:** Install the "mypy" extension for real-time type checking.

By configuring your IDE with these tools, you ensure consistent code quality and adhere to the project's coding standards effortlessly. 

If you prefer to do these steps manually, please read the following steps, if not, you can ignore this section.

### Code Formatting

We use **Black** for code formatting. To format your code, run:

```bash
black .
```

### Linting

**isort** is used for sorting imports. To sort imports, execute:

```bash
isort .
```

### Type Checking

**mypy** is employed for static type checking. To check types, run:

```bash
mypy torchebm/
```

### Testing

We utilize **pytest** for testing. To run tests, execute:

```bash
pytest
```

For test coverage, use:

```bash
pytest --cov=torchebm
```

## Documentation

### Docstring Conventions

All docstrings should adhere to the Google style guide. For example:

```python
def function_name(param1, param2):
    """Short description of the function.

    Longer description if needed.

    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.

    Returns:
        type: Description of return value.

    Raises:
        ExceptionType: Explanation of when this exception is raised.

    Examples:
        result = function_name(1, "test")
    """
    # Function implementation
```

### API Documentation Generation

The API documentation is automatically generated from docstrings using `generate_api_docs.py`, MkDocs, and the MkDocstrings plugin.

To update the API documentation:

1. Run the API documentation generator script:

    ```bash
    python gen_ref_pages.py
    ```

2. Build the documentation to preview changes:

    ```bash
    mkdocs serve
    ```

## Contributing

We welcome contributions! Please follow the guidelines below.

### Branching Strategy

- **main**: Contains stable code.
- **feature/branch-name**: For developing new features.
- **bugfix/branch-name**: For fixing bugs.

### Commit Messages

Use clear and concise commit messages. Follow the format:

```
Subject line (imperative, 50 characters or less)

Optional detailed description, wrapped at 72 characters.
```

### Pull Requests

Before submitting a pull request:

1. Ensure all tests pass.
2. Update documentation if applicable.
3. Adhere to code style guidelines.

## Additional Resources

- [Python's PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Black Code Formatter](https://black.readthedocs.io/en/stable/)
- [pytest Documentation](https://docs.pytest.org/en/stable/)

By following this guide, you will help maintain the quality and consistency of the `torchebm` library. Happy coding! 