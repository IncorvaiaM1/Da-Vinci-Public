# QILA Da Vinci

**Quantum-Inspired Local AI** — a from-scratch transformer architecture designed for sub-100MB deployment on CPU, with sub-millisecond per-token latency.

> ⚠️ This repository is intentionally sparse. The project is under active development and the core implementation is proprietary. This README exists to document the research direction and confirm the project is real and in progress.

---

## What This Is

QILA Da Vinci applies physics-native mathematics — specifically **Matrix Product States (MPS)** and tensor network theory from quantum many-body physics — to redesign the transformer attention mechanism from first principles.

This is **not** post-hoc compression of a classical transformer. The architecture treats attention itself as a tensor network contraction, enabling:

- A dramatically smaller parameter count (~15–50M)
- Near-lossless expressivity within the regime relevant to the task
- A sub-100MB total footprint suitable for edge/CPU inference
- Principled, physics-motivated control over model capacity via bond dimension χ

The core insight is that the entanglement structure of natural language imposes an **area law** on the information content of attention — the same law that makes MPS so effective in quantum physics. QILA exploits this directly rather than approximating it after the fact.

---

## Current Status

**Phase 1 — Mathematical Prototype (complete)**

- MPS tensor contraction engine (NumPy) with multilinearity verified
- MPS-native attention layer (`MPSAttention`) with MPS linear projections (`MPSLinear`)
- ~72% parameter compression vs. equivalent dense attention at matched sequence length
- Cayley unitary transform layer maintaining orthogonality error ~10⁻⁶ through backpropagation
- CPU benchmark confirming favorable crossover vs. dense matmul at practical bond dimensions
- Entanglement entropy profiler for principled χ selection on target data distributions

**In Progress**

- Full math verification in JAX before any C++ implementation
- Scaling toward a full inference pipeline
- DMRG-style SVD analysis to determine optimal bond dimension χ for natural language
- Benchmarking against current SOTA tensor-based attention methods

---

## Why This Is Novel

Existing tensor network work in ML falls into two categories:

1. **Weight compression** — apply MPS to compress existing weight matrices after training
2. **Embedding methods** — use tensor products to enrich input representations

QILA Da Vinci is neither. It is a **native MPS attention mechanism** designed from scratch, where the contraction itself *is* the attention operation. No prior published work addresses this formulation.

---

## Background & Motivation

This project draws on:

- Tensor network methods from quantum many-body physics (MPS, DMRG, entanglement entropy)
- Transformer architecture design (attention, GQA, RoPE, SwiGLU, RMSNorm)
- On-device inference constraints (llama.cpp as a reference point for the C++ target)

The goal is a model that runs fully locally — no cloud, no GPU — with latency and size characteristics that make it viable for embedded and edge applications.

---

## Repository Structure

Most implementation files are private. What is publicly visible here reflects the project's scope and direction, not its implementation.

---

## Status

🔬 Active research — solo project  
📅 Started: 2025  
👤 Author: Michael (physicist)

---

*If you're a recruiter or collaborator and want to know more, feel free to reach out directly.*
