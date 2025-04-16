# Torchebm > Core > Base_energy_function

## Contents

### Classes

- [`AckleyEnergy`](classes/AckleyEnergy) - Energy function for the Ackley function.
- [`BaseEnergyFunction`](classes/BaseEnergyFunction) - Abstract base class for energy functions (Potential Energy \(E(x)\)).
- [`DoubleWellEnergy`](classes/DoubleWellEnergy) - Energy function for a double well potential. \( E(x) = h Σ(x²-1)² \) where h is the barrier height.
- [`GaussianEnergy`](classes/GaussianEnergy) - Energy function for a Gaussian distribution. \(E(x) = 0.5 (x-μ)ᵀ Σ⁻¹ (x-μ)\).
- [`HarmonicEnergy`](classes/HarmonicEnergy) - Energy function for a harmonic oscillator. \(E(x) = 0.5 \cdot n\_steps \cdot Σ(x²)\).
- [`RastriginEnergy`](classes/RastriginEnergy) - Energy function for the Rastrigin function.
- [`RosenbrockEnergy`](classes/RosenbrockEnergy) - Energy function for the Rosenbrock function. \(E(x) = (a-x₁)² + b·(x₂-x₁²)²\).

## API Reference

::: torchebm.core.base_energy_function
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
