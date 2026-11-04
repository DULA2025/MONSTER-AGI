# 👾 MONSTER-AGI

**Liquid Neural Dynamics Constrained by the Formally Verified Monster Group / A₂₄ Structure**

> Continuous liquid residuals subordinated to the discrete, machine-checked algebraic skeleton of the Monster group 𝕄.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-green.svg)]()
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)]()

---

## 🌌 Overview

This repository explores a new architectural direction for AI systems:

- **Continuous liquid residuals** provide adaptive capacity.
- **Certified algebraic invariants** from the Monster group 𝕄, Griess algebra, and Norton 2A-axis classes act as hard constraints.
- **Discrete τ / triality transitions** are the only permitted way to change Norton class.
- A **certificate interface** continuously validates that the system respects the formalized spectrum.

The work builds on earlier investigations into Liquid Neural Networks, the Prime Inertia Engine, and Lean 4 formalizations of Monster-related structures.

---

## 🔐 Key Architectural Invariants

| Invariant | Description |
|-----------|-------------|
| **Norton Classes** | Nine dihedral product classes with exact rational inner products: `1`, `1/4`, `0`, `13/256`, `1/64`, `1/32`, `1/128`, `3/128`, `5/256` |
| **Class-Aware Barrier** | Residual is softly attracted to the certified radius of its current Norton class |
| **Graph Coupling** | Sparse interactions between axes proportional to certified Griess inner products |
| **Prime Inertia Control** | Learnable, strictly positive modulation of barrier and interaction strengths |
| **Certificate Discipline** | Continuous validation against the formalized spectrum; violation rate is tracked |
| **Discrete τ Sequences** | Multi-step triality orbits that reassign classes; the system re-settles under new invariants |

---

## 🤖 Current Capabilities

- Multi-axis liquid residual evolution (tested up to 16 axes)
- Higher-fidelity eigenspace mock of the 2.𝔹 decomposition
- Sparse graph coupling (ring + chords)
- Bounded Prime Inertia Engine control layer
- Modular-arithmetic residual as external task
- End-to-end **MNIST training harness** that encodes images into axis states, evolves them under certified dynamics, and classifies while monitoring algebraic violation rate
- Clean, reusable `MonsterA24System` API

---

## 📁 Repository Structure

```
MONSTER-AGI/
├── monster_a24_system.py      # Modular core system
├── monster_a24_mnist.py       # MNIST training harness
├── README.md
└── (future) lean/             # Interface stubs for Lean-exported checkers
```

---

## 🚀 Quick Start

### Core System

```bash
pip install torch
python monster_a24_system.py
```

**Expected output:** 16/16 certificate passes and zero violation rate after free evolution and after multi-step τ sequences.

### MNIST Experiment

```bash
pip install torch torchvision
python monster_a24_mnist.py
```

The harness reports both classification accuracy and certificate violation rate.

---

## 🧠 Design Principles

1. **Algebraic skeleton first** — continuous dynamics may not invent transitions that violate the formalized Norton spectrum.
2. **Certificate observability** — every experiment measures invariant violation rate alongside task performance.
3. **Liquid residual inside shells** — adaptive continuous evolution is permitted only within the region allowed by the current Norton class.
4. **Discrete generators for class change** — τ and related formalized elements are the sole mechanism for changing class.
5. **Bounded control** — Prime Inertia Engine scales remain strictly positive and limited.

---

## 🔗 Relation to Prior Work

- Prime Inertia Engine (arithmetic viscosity, modular dynamics)
- Liquid Neural Network investigations
- Interactive Monster Group / CFSG Lean formalization explorer
- Lean 4 developments covering Griess algebra, 2A-axes, Baby Monster, triality, and simplicity

---

## 📊 Status

**Research prototype.**  
The continuous–discrete coupling is stable and certificate-preserving on the tasks tested so far.

Active directions:
- Scaling eigenspace fidelity
- Fuller Griess products
- Connecting the certificate interface to actual Lean-exported checkers

---

## 📜 Citation

If you use or build upon this architecture, please cite:

```
DULA2025, MONSTER-AGI,
https://github.com/DULA2025/MONSTER-AGI
```

---
# Monster A24 model


```How to run
Install once: pip install flask flask-cors
Put monster_a24_final.pt in the same folder as server.py.
Run: python server.py
Open digit_app.html in your browser.
Draw a digit and click Recognize.
The app uses your actual trained Monster A24 model (near-zero violation, ~98 % MNIST accuracy) for the predictions.

```

---

## 📄 License

To be determined. See repository for current terms.

---

<p align="center">
  <sub>Built with curiosity at the intersection of sporadic groups, liquid dynamics, and machine-checked mathematics.</sub><br>
  🤖 🧠 ⚛️ 🌌
</p>
