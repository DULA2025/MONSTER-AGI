```markdown
# MONSTER-AGI

**Liquid Neural Dynamics Constrained by the Formally Verified Monster Group / A₂₄ Structure**

A research prototype exploring a new architectural direction for AI systems: continuous liquid residuals subordinated to the discrete, machine-checked algebraic skeleton of the Monster group 𝕄, its Griess algebra, Norton 2A-axis classes, and the A₂₄ / Leech lattice hierarchy.

The core idea is that the richest sporadic symmetry available in mathematics can serve as a hard inductive bias and certified state machine, while liquid (continuous-time) dynamics provide adaptive residual capacity inside the allowed algebraic shells.

## Overview

This repository contains the progressive development of a multi-axis liquid architecture whose continuous evolution is constrained at every step by the certified Norton spectrum and Griess inner products. Discrete transitions (τ / triality) are the only mechanism permitted to change Norton class. A certificate interface continuously validates that the emitted inner-product signal remains within tolerance of the Lean-formalized values.

The work builds on earlier investigations into Liquid Neural Networks, the Prime Inertia Engine, and the Lean 4 formalization of Monster-related structures.

## Key Architectural Invariants

- **Norton classes** — nine dihedral product classes of 2A-involutions with exact rational inner products:
  - 1, 1/4, 0, 13/256, 1/64, 1/32, 1/128, 3/128, 5/256
- **Class-aware inertia barrier** — residual is softly attracted to the certified radius of its current Norton class.
- **Graph-structured coupling** — sparse interactions between axes proportional to certified Griess inner products.
- **Prime Inertia Engine control layer** — learnable, strictly positive modulation of barrier and interaction strengths.
- **Certificate discipline** — continuous validation against the formalized spectrum; violation rate is tracked.
- **Discrete τ sequences** — multi-step triality orbits that reassign classes; the system re-settles under the new invariants.

## Current Capabilities

- Multi-axis liquid residual evolution (tested up to 16 axes)
- Higher-fidelity eigenspace mock of the 2.𝔹 decomposition
- Sparse graph coupling (ring + chords)
- Bounded control layer
- Modular-arithmetic residual as external task
- End-to-end MNIST training harness that encodes images into axis states, evolves them under the certified dynamics, and classifies while monitoring algebraic violation rate
- Clean, reusable `MonsterA24System` API

## Repository Structure

```
MONSTER-AGI/
├── monster_a24_system.py      # Modular core system (graph coupling, certificates, control)
├── monster_a24_mnist.py       # MNIST training harness using the architecture as backbone
├── README.md
└── (future) lean/             # Interface stubs for Lean-exported checkers
```

## Quick Start

### Core System
```bash
pip install torch
python monster_a24_system.py
```

Expected output includes 16/16 certificate passes and zero violation rate after free evolution and after multi-step τ sequences.

### MNIST Experiment
```bash
pip install torch torchvision
python monster_a24_mnist.py
```

The harness reports both classification accuracy and certificate violation rate.

## Design Principles

1. **Algebraic skeleton first** — continuous dynamics may not invent transitions that violate the formalized Norton spectrum or Griess structure.
2. **Certificate observability** — every experiment measures invariant violation rate alongside task performance.
3. **Liquid residual inside shells** — adaptive continuous evolution is permitted only within the region allowed by the current Norton class.
4. **Discrete generators for class change** — τ and related formalized elements are the sole mechanism for changing class.
5. **Bounded control** — Prime Inertia Engine scales remain strictly positive and limited.

## Relation to Prior Work

- Prime Inertia Engine (arithmetic viscosity, modular dynamics)
- Liquid Neural Network investigations (Tao modular-arithmetic challenge and related analyses)
- Interactive Monster Group / CFSG Lean formalization explorer
- Lean 4 developments covering Griess algebra, 2A-axes, Baby Monster, triality, and simplicity

## Status

Research prototype. The continuous–discrete coupling is stable and certificate-preserving on the tasks tested so far. Scaling eigenspace fidelity, replacing synthetic interactions with fuller Griess products, and connecting the certificate interface to actual Lean-exported checkers are active directions.

## Citation

If you use or build upon this architecture, please cite:

```
DULA2025, MONSTER-AGI,
https://github.com/DULA2025/MONSTER-AGI
```

## License

To be determined. See repository for current terms.
```
