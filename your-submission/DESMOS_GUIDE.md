# Desmos Mathematical Models & Graphing Guide

This guide provides the exact formulas, equations, data tables, and step-by-step instructions to plot the audit's findings in **Desmos Graphing Calculator** ([desmos.com/calculator](https://www.desmos.com/calculator)).

You can also open the pre-built interactive Desmos app included in this repo:
📂 **[`your-submission/desmos_visualizer.html`](desmos_visualizer.html)** (Double click to open in any web browser).

---

## 📈 Graph 1: Part B1 & B2 — KV-Cache VRAM Footprint & Preemption Boundary

This graph proves the maximum concurrent sequences an NVIDIA L4 GPU can hold before preemption occurs.

### Equations to paste into Desmos:

1. **Total GPU Memory Function ($M(x)$ in GB as a function of concurrent 4096-token sequences $x$):**
   ```latex
   M(x) = 8.4 + 1.6 + 0.469768x
   ```
   *Explanation:* $8.4\text{ GB}$ (Model weights) + $1.6\text{ GB}$ (CUDA runtime overhead) + $0.469768\text{ GB}$ ($4096 \text{ tokens} \times 114,688\text{ bytes/token}$).

2. **Available VRAM Threshold ($y = 22.08\text{ GB}$):**
   ```latex
   y = 22.08
   ```
   *Explanation:* $0.92 \times 24\text{ GB}$ usable memory ceiling on NVIDIA L4.

3. **Total Physical GPU VRAM ($y = 24\text{ GB}$):**
   ```latex
   y = 24
   ```

4. **Maximum Safe Concurrency Limit Line ($x \approx 25.7$):**
   ```latex
   x = 25.7
   ```

5. **Observed Data Points from `bench_log.csv` (Batch vs Total Allocated VRAM):**
   ```latex
   P = [(4, 11.93), (8, 13.74), (16, 17.48), (24, 21.23), (32, 21.71), (48, 21.71)]
   ```
   - At $x = 24$: $VRAM = 21.23\text{ GB}$ ($93\%$ KV util, **0 preemptions**).
   - At $x = 32$: Memory overflows $22.08\text{ GB}$ ceiling $\rightarrow$ **7 preemptions**.
   - At $x = 48$: Memory severely throttled $\rightarrow$ **23 preemptions**.

---

## 📈 Graph 2: Part B2 & B3 — Serving Throughput vs Batch Size Anomaly

This graph reveals why the previous report's claim that *"longer prompts give better throughput"* and *"batch 48 delivers 3200 tok/s"* is false.

### Equations & Tables to paste into Desmos:

1. **Short Prompt Throughput Data (512 prompt + 256 gen):**
   ```latex
   S = [(1, 70.2), (2, 132.3), (4, 261.0), (8, 495.4), (16, 883.2), (32, 1489.6), (64, 2267.3)]
   ```
   *(Enable 'Lines' in Desmos to show the curve).*

2. **Long Prompt Throughput Data (3584 prompt + 512 gen):**
   ```latex
   L = [(4, 565.4), (8, 902.6), (16, 1311.4), (24, 1607.4), (32, 1384.0), (48, 1298.5)]
   ```
   *(Enable 'Lines' in Desmos to show the curve).*

3. **Report v0 Linear Extrapolation (The Fictional Myth):**
   ```latex
   y = 66.97x
   ```
   - Shows where the intern predicted $3,214\text{ tok/s}$ at batch 48 vs the actual measured **$1,298.5\text{ tok/s}$** collapse.

---

## 📈 Graph 3: Part A3 — Cross-Lingual Tokenizer Fertility (GPT-2 vs MuRIL)

This graph visualizes the token-cost disparity per sentence across languages:

### Data to paste into Desmos:

1. **GPT-2 Tokens / Sentence ($x$: 1=Eng, 2=Hin, 3=Kan, 4=Tam):**
   ```latex
   G = [(1, 26.8), (2, 171.4), (3, 281.1), (4, 305.5)]
   ```

2. **Indic-Aware MuRIL Tokens / Sentence:**
   ```latex
   M = [(1, 26.6), (2, 23.7), (3, 20.2), (4, 19.8)]
   ```

3. **Point Labels:**
   - $(1, 26.8)$ $\rightarrow$ `English (GPT-2: 26.8 | MuRIL: 26.6)`
   - $(2, 171.4)$ $\rightarrow$ `Hindi (GPT-2: 171.4 | MuRIL: 23.7)`
   - $(3, 281.1)$ $\rightarrow$ `Kannada (GPT-2: 281.1 | MuRIL: 20.2)`
   - $(4, 305.5)$ $\rightarrow$ `Tamil (GPT-2: 305.5 | MuRIL: 19.8)`

---

## 🚀 How to Present in Desmos During Live Defense

1. Open [desmos.com/calculator](https://www.desmos.com/calculator) on your browser during screen-share.
2. Paste the equations above for each section.
3. Use the intersection of $M(x) = 22.08$ to visually prove that the GPU physically runs out of memory at $x \approx 26\text{--}28$ sequences, perfectly explaining why preemption begins at batch 32.
