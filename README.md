# llama.cpp Concurrent CPU Inference Roadmap

This project aims to enable **efficient concurrent inference** using [llama.cpp](https://github.com/ggerganov/llama.cpp) on **CPU-only systems**, starting with the **Apple M1** as the first target. The final goal is to scale to high-concurrency environments such as **32-core Linux servers**, supporting real-world applications with 10–20 concurrent users.

---

## 🔍 Objective

Enable multiple simultaneous low-latency inferences on CPU by:
- Running multiple `llama_context`s in parallel.
- Efficiently scheduling token generation using cooperative multitasking.
- Optimizing memory layout and KV caching for CPU cache locality.
- Optionally exploring micro-batching and decode fusion for higher throughput.

---

## 🧭 Roadmap Overview

The roadmap is divided into six key stages:

### ✅ Stage 1: Baseline System
- [x] Load one `llama_model` and initialize multiple `llama_context`s.
- [x] Use a thread pool to manage multiple concurrent contexts.
- [x] Measure and benchmark initial throughput and latency.

### ⚙️ Stage 2: Cooperative Scheduler
- Implement a round-robin scheduler for job dispatching.
- Perform one token decode per step across active contexts.
- Introduce job queues and callbacks.

### 🧠 Stage 3: KV Cache Optimization
- Use `mmap` or fixed-size memory arenas.
- Optimize memory layout for cache efficiency (especially on M1).
- Profile memory access and reduce cache misses.

### 🚀 Stage 4: Micro-Batching (Experimental)
- Modify llama.cpp to support decoding multiple contexts in a batch.
- Investigate merge and split operations for KV/state tensors.
- Compare performance vs standard round-robin inference.

### 📊 Stage 5: Benchmarking & Real-World Simulation
- Build a test server (WebSocket or FastAPI).
- Simulate concurrent users with realistic prompts.
- Benchmark latency, memory usage, and throughput.

### 📦 Stage 6: Port to High-Core Linux
- Deploy system on 32-core Linux machine.
- Use CPU affinity and NUMA-aware memory optimizations.
- Re-run benchmarks and evaluate scale-up behavior.

---

## 📁 Components

- `llama_utils.cpp`: Handles model loading and context creation.
- `threadPool.cpp`: Manages a fixed pool of llama contexts.
- `scheduler.cpp`: Dispatches inference jobs cooperatively.
- `inference.cpp`: Handles per-token generation loop.
- `main.cpp`: Example/testing harness.

---

## 🛠️ Build & Export

- Build shared library `libllama_concurrent.so`
- Export async inference API: `llama_infer_async(prompt, callback)`
- Bindings planned for:
  - Python (via `ctypes` or `cffi`)
  - Rust (via `bindgen` or manual FFI)

---

## 📈 Progress Tracker

A [Google Sheet tracker](https://docs.google.com/spreadsheets/d/1TaWbsxEkF9VHrNGANHSYGqfrF0lsnZr4vnZvgDtgGSc/edit?usp=sharing) (or local Excel version) is used to track progress across each task and stage.
