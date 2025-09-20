# Phase 3 / Phase 4 — Fair Simulator Test Suite & Hardware-Validation Playbook

**Purpose:** Give a reproducible, unbiased playbook to compare TTA with mainstream architectures (RISC/ARM, x86, GPU), and to decide whether to proceed to FPGA/ASIC validation (Phase 4).  
**Scope:** Kernel selection, metrics, experiment matrix, configs, analysis, and decision gates.

---

## Table of contents
- [Goals & Decision Criteria](#goals--decision-criteria)  
- [High-level approach](#high-level-approach)  
- [Kernel suite (must-run list)](#kernel-suite-must-run-list)  
- [Metrics & measurement methodology](#metrics--measurement-methodology)  
- [Experiment matrix & configurations](#experiment-matrix--configurations)  
- [Simulator requirements & constraints](#simulator-requirements--constraints)  
- [Reproducibility checklist](#reproducibility-checklist)  
- [Data collection format](#data-collection-format)  
- [Analysis recipe & example plots](#analysis-recipe--example-plots)  
- [Decision gates — when to move to FPGA/ASIC (Phase 4)](#decision-gates---when-to-move-to-fpgaasic-phase-4)  
- [FPGA validation quick roadmap](#fpga-validation-quick-roadmap)  
- [Appendix: Example configs & commands](#appendix-example-configs--commands)

---

## Goals & Decision Criteria

**Primary goal:** Determine if TTA yields *meaningful* energy/performance advantages on representative AI kernels compared to ARM/RISC, x86, and GPU baselines.

**Target thresholds (for considering hardware):**
- *Minimum viable:* reproducible **≥15–20%** energy efficiency improvement (ops/J) on multiple AI kernels vs ARM baseline, **without** >2× area/complexity penalty.
- *Compelling / move-to-FPGA threshold:* **≥30%** energy improvement on key kernels + good scalability.
- *Breakthrough threshold:* **≥2×** energy improvement or unique algorithmic benefit (e.g., energy-efficient attention primitive) that compilers/ISAs can’t reasonably match.

**Secondary factors** to weigh: code density, scheduler complexity, compiler required sophistication, FPGA resource utilization estimates.

---

## High-level approach

1. Implement a **cycle-accurate TTA simulator** (done in Phase 1).  
2. Implement or adopt **cycle-accurate simulators** representing:
   - RISC baseline (ARM-like microarchitecture)
   - x86 baseline (simplified but realistic micro-ops model)
   - GPU baseline (SIMT model or a simplified many-core model)
3. Standardize **memory hierarchy, precision formats, and problem sizes** across runs.
4. Run the same kernels on each simulator with identical input data and seeds.
5. Collect energy, cycles, instruction/move counts, and utilization data.
6. Analyze per-kernel and aggregated results; apply decision gates.

---

## Kernel suite (must-run list)

Start small, expand later.

### Scalar & micro
- `axpb` — `y = a*x + b` (scalar)
- Polynomial evaluation (Horner)

### Vector
- `dot_product` — small and medium sizes (N=256, 4096, 65536)
- `AXPY` — `y = a*x + y`

### Matrix
- `GEMV` (matrix-vector multiply) — dense
- `small GEMM` (e.g., 64×64, 128×128)

### Convolution & Patches
- 2D convolution patches for small kernels (3×3, 5×5) and varying stride

### AI primitives
- Attention (scaled dot-product attention; QK^T + softmax, reduce)
- MHA (multi-head attention) simplified variant
- LayerNorm / BatchNorm micro-benchmarks
- Activation functions (ReLU, GELU) and quantized variants

### Memory patterns
- Streaming (sequential)
- Gather/scatter (sparse access)
- Reduction (tree reduction patterns)

**Variants:** For each kernel, test `float32`, `float16` (where applicable), `int8`. Test sparse vs dense inputs for attention-like kernels.

---

## Metrics & measurement methodology

**Primary metrics**
- **Energy Efficiency:** `ops_per_joule` (or joules per operation). For AI: TOPs/W or operations/J
- **Throughput / Performance:** cycles and effective ops/sec
- **Latency:** wall-clock cycles for small batch sizes
- **Code Density:** number of encoded instructions / moves per kernel
- **Utilization:** functional unit and bus utilization (%)
- **Memory traffic:** bytes read/written to each level (register, scratchpad, DRAM)

**Secondary metrics**
- **Scheduler complexity:** size or runtime of scheduling step (compile-time)
- **Simulator runtime** (for reproducibility)
- **FPGA resource estimate** (LUTs, BRAM, DSPs) — derived later

**Measurement rules**
- Use **cycle-accurate** models with the **same memory model** (registers/scratchpad/cache/DRAM) parameters across architecture sims.
- Energy model must be componentized: per-op FU cost + transport/bus cost + memory access cost. Use the same per-technology cost assumptions across archs (configurable file).
- Each run must be repeated **≥5 times** and report median ± std. Use identical input sets and RNG seeds.

---

## Experiment matrix & configurations

Design an experiment matrix with the following axes:

- Architecture: `TTA`, `RISC-ARM`, `x86`, `GPU`
- Kernel: as above
- Precision: `FP32`, `FP16`, `INT8`
- Problem size: small / medium / large
- Memory pattern: dense / sparse
- Scheduling mode (TTA only): static schedule / heuristic / auto-scheduler (if available)
- Parallelism (GPU): single-thread / full-warp / multi-warp

**Example minimal matrix entry**
```
{arch: "TTA", kernel: "dot_product", precision: "FP32", N: 4096, memory: "dense"}
```

Run *all* combinations in a reproducible batch script.

---

## Simulator requirements & constraints

- **Cycle-accurate timing** for compute/memory/transport.
- Pluggable **energy cost tables** (`energy_costs.toml`) with:
  - per-FU: energy/op
  - per-move: energy/bit/mm (or per-bus-use param)
  - memory-level costs: register / scratchpad / L1 / DRAM
- **Same instruction/operation semantics** for kernels across simulators (numerical correctness).
- Provide hooks to export:
  - cycles, ops count, energy breakdown (compute vs transport vs memory), utilization, and traces (partial).
- **Config files** for each run (so experiments are reproducible).

---

## Reproducibility checklist

- [ ] Pin simulator versions (commit SHA or version tag).
- [ ] Pin toolchain (Rust/Cargo/Python versions) and containerize (Docker) or provide Nix/Cross files.
- [ ] Fix RNG seeds and any non-deterministic behavior.
- [ ] Provide exact kernel inputs (store sample datasets).
- [ ] Provide exact `energy_costs.toml` and `tta.toml` used for run.
- [ ] Log system and hardware used for simulation (host CPU, RAM).
- [ ] Output raw traces and aggregated CSVs.

---

## Data collection format

Store one CSV per run with following columns (example):

```
run_id,arch,kernel,precision,size,problem_variant,cycles,ops,energy_compute_j,energy_transport_j,energy_memory_j,energy_total_j,fu_util_pct,bus_util_pct,code_density,compiled_sched_time_s,seed,timestamp
```

Also store a JSON manifest with hyperparameters and config references.

---

## Analysis recipe & example plots

**Analysis steps**
1. Normalize `ops` across architectures (count mathematical ops—multiply-adds counted as 2 ops or according to convention).
2. Compute `ops_per_joule = ops / energy_total_j`.
3. Compute performance-per-watt and energy-per-op.
4. Aggregate by kernel and by architecture.
5. Plot:
   - `ops_per_joule` bar chart per kernel (arch side-by-side).
   - Energy breakdown stacked bars (compute vs transport vs memory).
   - Utilization heatmaps (FU/bus vs time or kernel).
   - Code density comparison (instructions/moves).

**Key comparisons**
- TTA vs ARM baseline (primary).
- TTA vs GPU for each kernel (important for large-batch cases).
- Scaling plots: energy_efficiency vs problem_size.

**Quick statistical checks**
- Significance test (paired t-test or non-parametric) across repeated runs to ensure results are robust.

---

## Decision gates — when to move to FPGA/ASIC (Phase 4)

**Gate A — “Worth FPGA”**
- TTA shows **≥30%** ops_per_joule improvement vs ARM on at least **2** representative AI kernels (e.g., GEMV + Attention).
- The improvement remains when using conservative (pessimistic) energy model parameters.
- Scheduler runtime and complexity are manageable (compile time ≤ some threshold) or automatable.

**Gate B — “Worth ASIC”**
- FPGA prototype confirms energy advantage within **±20%** of simulator prediction.
- FPGA resource use (LUTs, DSPs, BRAM) is within practical limits for target device family.
- There is a plausible commercial or research use-case (edge inference, low-power server), and a software path (compiler/toolchain) is feasible.

If Gate A fails: stop and prepare a paper/technical note exploring why. If Gate A passes but FPGA fails to validate, re-evaluate energy model for errors.

---

## FPGA validation quick roadmap

1. **Microbenchmark FPGA core** — small specialized TTA FU (multiplier, adder) to measure actual energy/per-op.
2. **Implement a TTA core** on FPGA (small config) — include scratchpad/reg file, bus, and a MAC unit.
3. **Run identical kernel(s)** on FPGA (same inputs) and measure:
   - Power draw (on-board PMBus or external power meter)
   - Throughput and latency
4. **Compare** FPGA energy numbers to simulator predictions. Recalibrate energy model accordingly.
5. **Scale up** (multiple FUs) and measure area/perf/power tradeoffs.
6. If validated, prepare for ASIC (only if gains are compelling).

**Measurement note:** On FPGA, power measurements must isolate the FPGA static power vs active switching. Use idle baseline subtraction and test windows.

---

## Appendix: Example configs & commands

### Example `energy_costs.toml` (snippet)
```toml
[functional_units]
mul = {energy_pj = 50e-12}   # 50 pJ per multiply (example)
add = {energy_pj = 20e-12}
mac = {energy_pj = 60e-12}

[transport]
per_move_pj = 5e-12          # per move cost (configurable)

[memory]
register_r = 0.1e-12
scratchpad_access = 2e-12
dram_access = 100e-12
```

### Example run command
```bash
# Run a matrix-vector test on TTA simulator with config
./tta_sim --kernel gemv --size 1024 --precision fp16 --config config/tta.toml --energy config/energy_costs.toml --seed 42 --out results/tta_gemv_1024_fp16.csv
```

### Example analysis pseudocode (Python-like)
```py
import pandas as pd
df = pd.read_csv("results_all.csv")
df['ops_per_joule'] = df.ops / df.energy_total_j
agg = df.groupby(['arch','kernel'])['ops_per_joule'].median().unstack()
agg.plot.bar()
```

---

## Practical notes & tips

- **Equal footing:** Make decisions conservatively—use a pessimistic energy model for TTA (so you’re not surprised in FPGA).
- **Scheduler automation:** Invest early in an automated TTA scheduler. Manual scheduling won’t scale and will skew results.
- **Open-source path:** If validated, publish the reference TTA core and simulator configs (like RISC-V did). Community uptake can help ecosystem tooling.
- **Record everything:** Save configs, seeds, and host metadata; future reviewers will need it to trust the results.

---

## Closing checklist (what to produce & commit for Phase 3)
- [ ] Simulator binaries and commit SHAs for all architectures  
- [ ] `energy_costs.toml` and `tta.toml` used for runs  
- [ ] Kernel input datasets and seed file  
- [ ] Batch-run scripts for full experiment matrix  
- [ ] Raw CSV outputs + JSON manifests  
- [ ] Analysis notebook (plots, aggregated tables)  
- [ ] Decision summary document (include Gate A/B statuses)
