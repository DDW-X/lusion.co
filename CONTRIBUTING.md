# Contributing to the lusion.co Reverse Engineering Project

Thank you for your interest in contributing to this reverse engineering research repository. All contributions must maintain the highest standards of graphics systems engineering, mathematical rigor, and runtime performance.

---

## 1. Architectural Principles & Performance Invariants

Any proposed pull request containing runtime code, shader modules, or engine modifications must strictly adhere to the following low-level engineering invariants:

### A. Zero-Allocation Frame Loop Invariant (Strict No-GC Rule)
* **No `new` Operator in Frame Loops**: Instantiating objects, vectors, matrices, or closures inside `requestAnimationFrame`, pointer listeners, wheel callbacks, or simulation loops is strictly prohibited.
* **Static Module Scratchpads**: Re-use pre-allocated module registers (e.g., `_v1`, `_v2`, `_m0`, `_sphere$4`, `_ray$3`).
* **Zero GC Pressure**: Code must demonstrate **$0.00\text{ KB/frame}$** nursery heap allocation in Chrome DevTools Memory Timeline.

### B. V8 Hidden Class Monomorphism & Element Kinds
* **Class Property Stability**: All object properties must be declared within constructor definitions. Never dynamically add (`obj.newProp = val`) or delete (`delete obj.prop`) properties, as this transitions V8 Maps to dictionary mode.
* **Preserve `PACKED_ELEMENTS`**: When manipulating arrays, avoid creating sparse arrays or holes (`arr[100] = x` when `length == 2`). Keep arrays dense to allow TurboFan to compile direct memory offsets.
* **TypedArray Views over Copies**: Never call `.slice()` on ArrayBuffers during frame loops. Use zero-copy `.subarray()` views.

### C. Mathematical Rigor & LaTeX Derivations
* Any PR adding or modifying physics simulations, post-processing optical kernels, coordinate projections, or non-Euclidean transformations **must include full first-principles mathematical derivations in LaTeX** within documentation.
* Include proofs for energy conservation, divergence-free conditions, or conformal angle preservation where applicable.

---

## 2. Coding Standards

* **GLSL Shader Architecture**:
  * Avoid dynamic branching inside loops (`for` loops must have constant or unrollable bounds).
  * Use vectorized ALU operations over scalar operations (`dot()`, `cross()`, `mix()`, `smoothstep()`).
  * Prefer single-pass fused uber-shaders over multi-pass render targets where bandwidth is critical.
* **File & Directory Structure**:
  * Core documentation: [`README.md`](README.md).
  * Research logs & telemetry: `walkthrough.md`.
  * Automation scripts: `scripts/`.

---

## 3. Pull Request Submission Checklist

Before opening a pull request, ensure you have:
1. Verified that the application builds and runs at a sustained **120 FPS** on local testing environments.
2. Verified zero minor GC pause jitter via Chrome DevTools Performance profiler.
3. Provided clean mathematical documentation with LaTeX formulas.
4. Ensured all commits follow standard Conventional Commits formatting (`feat:`, `fix:`, `docs:`, `perf:`).
5. Signed off on research ethics and responsible disclosure guidelines.
