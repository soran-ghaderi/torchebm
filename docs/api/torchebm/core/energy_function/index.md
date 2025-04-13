# Torchebm > Core > Energy_function

## Contents

### Classes

- [`AckleyEnergy`](classes/AckleyEnergy) - Energy function for the Ackley function.
- [`DoubleWellEnergy`](classes/DoubleWellEnergy) - Energy function for a double well potential. E(x) = h * Σ((x²-1)²) where h is the barrier height.
- [`BaseEnergyFunction`](classes/EnergyFunction) - Abstract base class for energy functions (Potential Energy E(x)).
- [`GaussianEnergy`](classes/GaussianEnergy) - Energy function for a Gaussian distribution. E(x) = 0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ).
- [`HarmonicEnergy`](classes/HarmonicEnergy) - Energy function for a harmonic oscillator. E(x) = 0.5 * k * Σ(x²).
- [`RastriginEnergy`](classes/RastriginEnergy) - Energy function for the Rastrigin function.
- [`RosenbrockEnergy`](classes/RosenbrockEnergy) - Energy function for the Rosenbrock function. E(x) = (a-x₁)² + b·(x₂-x₁²)².

## API Reference

::: torchebm.core.energy_function
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_source: true
      show_symbol_type_heading: true
      show_symbol_type_toc: true
      show_docstring_attributes: false
      show_docstring_classes: true
      show_docstring_functions: true
      trim_doctest_flags: true
      show_category_heading: false
      show_if_no_docstring: true
      members_order: source
      show_signature_annotations: true
      separate_signature: true
      unwrap_annotated: true
      docstring_section_style: table
      inherited_members: false
      members:
        - "!__*"
