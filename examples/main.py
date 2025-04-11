#!/usr/bin/env python
"""
Main entry point for running TorchEBM examples.
"""

import argparse
import os
import importlib
import sys

def list_examples():
    """List all available examples by category."""
    categories = {
        'core': {
            'energy_functions': ['landscape_2d', 'multimodal', 'parametric']
        },
        'samplers': {
            'langevin': ['gaussian_sampling', 'multimodal_sampling', 'visualization_trajectory', 'advanced'],
            'hmc': ['gaussian_sampling', 'advanced', 'mass_matrix']
        },
        'visualization': {
            'basic': ['contour_plots', 'distribution_comparison'],
            'advanced': ['trajectory_animation', 'parallel_chains', 'energy_over_time']
        },
        'utils': ['performance_benchmark', 'convergence_diagnostics']
    }
    
    print("\nAvailable TorchEBM Examples:")
    print("============================\n")
    
    for category, subcategories in categories.items():
        print(f"{category.upper()}:")
        if isinstance(subcategories, dict):
            for subcategory, examples in subcategories.items():
                print(f"  {subcategory.capitalize()}:")
                for example in examples:
                    print(f"    - {category}/{subcategory}/{example}")
                print()
        else:
            for example in subcategories:
                print(f"  - {category}/{example}")
            print()

def run_example(example_path):
    """Run a specific example."""
    try:
        # Extract the module path
        module_path = example_path.replace('/', '.')
        
        # Try to import and run the example
        module = importlib.import_module(f"examples.{module_path}")
        print(f"Successfully imported example: {example_path}")
        
        # If the module has a main function, call it
        if hasattr(module, 'main'):
            module.main()
        
        return True
    except ImportError as e:
        print(f"ERROR: Could not import example '{example_path}'")
        print(f"  {str(e)}")
        return False
    except Exception as e:
        print(f"ERROR: An error occurred while running '{example_path}':")
        print(f"  {type(e).__name__}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run TorchEBM examples')
    parser.add_argument('example', nargs='?', help='Example to run (e.g., samplers/langevin/gaussian_sampling)')
    parser.add_argument('--list', '-l', action='store_true', help='List all available examples')
    
    args = parser.parse_args()
    
    if args.list or not args.example:
        list_examples()
        if not args.example:
            print("Use --help for more information or specify an example to run.")
        return
    
    success = run_example(args.example)
    if not success:
        print("\nTry running with --list to see all available examples.")

if __name__ == "__main__":
    # Add the parent directory to sys.path to allow importing modules
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main() 