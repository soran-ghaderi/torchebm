# API Documentation Generation

The API documentation for TorchEBM is automatically generated from docstrings in the code using `generate_api_docs.py` (specifically implemented for TorchEBM), MkDocs and the MkDocstrings plugin.

## How to Update API Documentation

Whenever you make changes to the codebase that affect the API (adding new modules, functions, classes, or updating docstrings), you should regenerate the API documentation:

1. Run the API documentation generator script:

```bash
chmod +x generate_api_docs.py
./generate_api_docs.py
```

2. If the script added new modules, you might need to update the `nav` section in your `mkdocs.yml` file. The script will provide the necessary configuration.

3. Build the documentation to preview your changes:

```bash
mkdocs serve
```

## Documentation Style

All docstrings in TorchEBM should follow the Google style guide. For example:

```python
def function_name(param1, param2):
    """Short description of the function.
    
    Longer description if needed.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
        
    Returns:
        type: Description of return value
        
    Raises:
        ExceptionType: When and why this exception is raised
        
    Examples:
        ```python
        result = function_name(1, "test")
        ```
    """
    # Function implementation
```