# RRI → C + OpenMP + OpenCL Conversion Plan

Status: **milestones 1-8 implemented, unit-tested, and validated against
the compiled Fortran reference on the real solo30s dataset over its full
360-hour run (<0.4% relative error throughout, including the flood
peak) -- on BOTH the CPU/OpenMP path and the OpenCL path, the latter
confirmed on a real GPU (AMD Polaris10 via Mesa Clover, OpenCL 1.1), not
just PoCL. See README.md's "OpenCL backend" section for the kernel
extraction, cross-backend validation, remote-GPU build/run procedure,
and honest timing (GPU currently slower than 32-core OpenMP on this
problem size -- expected, see that section; this milestone's job was
correctness, not throughput). Milestone 9 (TSAS) and dam/diversion/
boundary/custom-cross-section support remain undone.**

An initial full-length CPU-path validation pass found a large divergence
(up to 57% on storage.dat, ~53% at the flood peak) that has since been
root-caused (a missing river-channel bed-incision offset, `zb` vs
`zb_riv` conflated -- see README.md's "Root-caused bug" and "Porting
gotchas" sections) and fixed; the numbers above are post-fix. This
document remains the spec for any remaining gaps.

Reference sources (read-only, do not modify):

- `$HOME/dev/RRI_1.4.2.7_Linux/` — canonical Fortran source (v1.4.2.7). Root
  `.f90` files are the reference; `openmp/` and `gpuoffload/` subdirs are
  upstream's own (minimal — only `funcr`'s main loop in `RRI_Riv.f90`)
  parallelization experiments, useful as a hint but not authoritative.
- `$HOME/dev/RRIpy/` — the Python port (in progress as of this writing) —
  useful as a second, more readable reference once it's finished, and its
  `tests/` fixtures (tiny synthetic domains) should be reused for
  cross-validation.

## 1. Goals

1. A C11 implementation, numerically faithful to the Fortran reference
   (same physics, same adaptive RK45 river/slope routing, same Green-Ampt
   infiltration / groundwater exchange / dam operation rules).
2. OpenMP parallelism on the CPU for all embarrassingly-parallel per-cell
   loops.
3. An OpenCL backend offering the same kernels for GPU execution, selected
   at build time or runtime.
4. Validated against the Fortran (and/or finished Python) reference on
   small synthetic domains before ever trusting it on real data.

Non-goals for v1: MPI/multi-node, mixed precision, GUI, TSAS particle
tracking (`RRI_TSAS.f90`) — port that last, after the core hydraulic loop
is solid, since it's a diagnostic/tracer feature layered on top of already-
computed flows and not needed for a first working GPU port.

## 2. Why this model parallelizes well

The Fortran code already stores river and slope state in **compressed
1D index arrays** (`riv_idx`, `slo_idx`), built once at startup by
`riv_idx_setting` / `slo_idx_setting` (`RRI_Sub.f90`). Each active
river/slope cell `k` carries precomputed neighbor lookups:

- `down_riv_idx(k)`, `dis_riv_idx(k)`, `zb_riv_idx(k)` — single downstream
  neighbor + distance + bed elevation (river: strictly a tree/DAG, one
  outflow direction per cell).
- `down_slo_idx(l, k)`, `dis_slo_idx(l, k)`, `len_slo_idx(l, k)` for
  `l = 1..lmax` (4 or 8 directions) — slope cells can exchange with up to
  8 neighbors.

This is exactly a **CSR-style sparse graph** representation, and the hot
loops (`funcr`, `qr_calc`, `qs_calc`/`funcs`, `funcg`, `storage_calc`,
`hr2vr`/`vr2hr`, `evp`, `infilt`) are **per-`k` independent** given the
previous timestep's state — a textbook `parallel_for` over cells. This
maps directly onto both an OpenMP `#pragma omp parallel for` and an OpenCL
`clEnqueueNDRangeKernel` over the cell index. Keep the Fortran's
struct-of-arrays (SoA) layout — do not switch to array-of-structs, it
would hurt GPU memory coalescing for no benefit.

What does **not** parallelize over cells, and must stay serial/host-side:

- The adaptive RK45 step-size control loop itself (`RRI.f90`'s main time
  loop: compute error norm across the whole domain, accept/reject/shrink
  `ddt`, retry) — a reduction (max error) after each parallel stage, then
  a scalar decision. This is a classic "parallel kernel + host-side
  reduction + host-side control flow" pattern, not a bottleneck to
  parallelize further.
- Dam operation logic (`RRI_Dam.f90`: `dam_checkstate`, `dam_write*`) —
  small count of dams, inherently serial/branchy, leave on host.
- Boundary/diversion bookkeeping (`RRI_Bound.f90`, `RRI_Div.f90`) — small,
  index-scatter, host-side.
- Output/IO (`RRI_Tecout.f90`, ascii/binary grid writers) — host-side,
  overlap with compute via double-buffering later if profiling justifies
  it, not a v1 concern.

## 3. Target module layout

Mirror the Fortran module boundaries 1:1 so the port stays auditable
against the reference, one file at a time:

```
RRI.opencl/
├── CMakeLists.txt
├── PLAN.md                    (this file)
├── include/
│   └── rri/
│       ├── globals.h          # struct rri_state — replaces RRI_Mod.f90 / RRI_Mod2.f90 module vars
│       ├── dam.h               # RRI_Mod_Dam.f90 struct
│       ├── tecout.h            # RRI_Mod_Tecout.f90 struct
│       ├── backend.h           # backend-selection macros (OMP_BACKEND / OPENCL_BACKEND)
│       └── kernels.h           # shared per-cell kernel signatures (C and .cl both include a
│                                 common .h via -DKERNEL_LANG so the math body is written ONCE)
├── src/
│   ├── main.c                  # RRI.f90 driver — STEP 0/1/2 + main time loop
│   ├── rri_read.c              # RRI_Read.f90 — config parsing (mirror RRI_Input.py's
│   │                             hardcoded-vs-parsed choice: decide in step 5 below)
│   ├── rri_sub.c               # RRI_Sub.f90 — read_gis_int/real, idx_setting, ij2idx/idx2ij,
│   │                             storage_calc, hubeny_sub
│   ├── rri_section.c           # RRI_Section.f90 — set_section, hr2vr, vr2hr, sec_hq_riv
│   ├── rri_riv.c / .cl         # RRI_Riv.f90 — funcr, qr_calc, hq_riv
│   ├── rri_rivslo.c / .cl      # RRI_RivSlo.f90 — funcrs (river/slope exchange)
│   ├── rri_slope.c / .cl       # RRI_Slope.f90 — funcs, qs_calc
│   ├── rri_gw.c / .cl          # RRI_GW.f90 — funcg, qg_calc, hg_calc, gw_recharge/lose/exfilt
│   ├── rri_infilt.c / .cl      # RRI_Infilt.f90 — Green-Ampt infiltration
│   ├── rri_evp.c / .cl         # RRI_Evp.f90 — evaporation
│   ├── rri_dam.c                # RRI_Dam.f90 — host-side only
│   ├── rri_div.c                # RRI_Div.f90 — host-side only
│   ├── rri_bound.c              # RRI_Bound.f90 — host-side only
│   ├── rri_tecout.c             # RRI_Tecout.f90 — host-side only
│   └── rri_tsas.c               # RRI_TSAS.f90 — v2, after core loop is validated
├── cl/
│   └── *.cl                     # generated/compiled from src/rri_*.cl bodies (see §6)
├── tests/
│   ├── unit/                    # per-kernel numeric tests (CPU reference vs OpenMP vs OpenCL)
│   └── fixtures/                # tiny synthetic ESRI ASCII grids (reuse RRIpy's tests/fixtures
│                                  once that port is done, keep both repos' fixtures identical)
└── third_party/                 # none expected; keep this section empty unless a real need appears
```

## 4. Data structures (`globals.h`)

Direct C translation of the Fortran module-level arrays in `RRI_Mod.f90` /
`RRI_Mod2.f90`, grouped into one `struct rri_state` passed by pointer to
every kernel (no globals — needed for OpenCL portability and for running
CPU/GPU reference comparisons side by side in the same process):

```c
typedef struct {
    int ny, nx;
    double xllcorner, yllcorner, cellsize;
    /* grid arrays, ny*nx, row-major, matching Fortran's (i,j) order */
    double *zb, *acc; int *domain, *riv, *land, *dir;
    ...
} rri_grid;

typedef struct {
    int count;                 /* riv_count or slo_count */
    int   *down;                /* down_riv_idx / down_slo_idx flattened lmax*count */
    double *dis, *len, *zb;
    double *ns, *width, *depth, *height, *area_ratio;  /* river */
    double *ka, *da, *dm, *beta, *soildepth, *gammaa;   /* slope, infiltration */
    ...
} rri_cellset;   /* one instance for river, one for slope */

typedef struct {
    rri_grid grid;
    rri_cellset riv, slo;
    rri_dam_state dam;
    double time, ddt;
    ...
} rri_state;
```

Allocate as flat `double*`/`int*` buffers (not arrays-of-structs) so the
same pointers hand off directly to `clCreateBuffer` in the OpenCL backend
without a repacking step.

## 5. Design decisions to make explicit before coding (flag here, decide in review)

1. **Config format**: Fortran reads a runtime `RRI_Input.txt`-style
   namelist (`RRI_Read.f90`). RRIpy's author instead hardcoded config as
   a Python module (`RRI_Input.py`) — a deliberate simplification for
   that port. For the C port, prefer reading the *original* Fortran-style
   text config at runtime (a small hand-written parser, `rri_read.c`) —
   C has no reason to inherit the Python shortcut, and a runtime config
   is more useful for a systems port meant to run many scenarios/GPUs.
2. **Precision**: Fortran uses `real(8)` throughout. OpenCL double support
   is not guaranteed on all devices — require `cl_khr_fp64` at device
   selection time and fail loudly (per this user's GRASS convention: fail
   loudly rather than silently downgrading precision) if unavailable,
   rather than silently falling back to `float`.
3. **RK45 coefficients**: `RRI.f90`'s main loop uses embedded
   Runge-Kutta-Fehlberg (Cash-Karp-style, given the `b31..b65`, `c1,c3,c4,c6`,
   `dc1..dc6` coefficient names and "stepsize underflow" error) for both
   river (`vr`/volume-based) and slope (`hs`/head-based) routing, with
   independent adaptive `ddt` for river and slope. Port these coefficients
   verbatim from `RRI.f90` (search `b21`, `b31`, `c1`, `dc1` in that file) —
   do not re-derive or substitute a different embedded RK pair.
4. **Boundary/dam/div coupling**: these mutate cellset state read by the
   next parallel stage. Keep them as explicit host-side sync points
   between kernel launches (matches how the Fortran itself sequences
   `call read_bound` / `call dam_checkstate` between RK stages) — do not
   try to fuse them into the GPU kernels.
5. **OpenMP/OpenCL code duplication**: avoid writing the same physics
   twice. Write each kernel's math body once in a header
   (`kernels.h`, e.g. `RRI_FUNCR_BODY(k, ...)` as a macro or a `static
   inline` function marked `#ifdef __OPENCL_VERSION__`-portable — no
   pointers-to-struct inside kernels, flat array args only, since OpenCL C
   is a restricted C dialect), and `#include` it both from `rri_riv.c`
   (wrapped in an `omp parallel for`) and from `rri_riv.cl` (wrapped in
   `__kernel` + global id lookup). This is the standard way to keep an
   OpenMP and OpenCL implementation from diverging.

## 6. Build system

CMake, two targets:

- `rri_cpu` — C11 + OpenMP, always built, this is the correctness
  reference and the fallback for machines without a usable OpenCL ICD.
- `rri_cl` — same `main.c` driver, links against OpenCL, dispatches the
  hot kernels via `.cl` sources embedded at build time (CMake
  `bin2c`-style embedding, so the binary is self-contained and doesn't
  need to locate `.cl` files at runtime).

Both targets share `rri_sub.c`, `rri_section.c`, `rri_dam.c`, `rri_div.c`,
`rri_bound.c`, `rri_tecout.c`, `rri_read.c` unchanged — only the hot
per-cell kernels (`rri_riv`, `rri_rivslo`, `rri_slope`, `rri_gw`,
`rri_infilt`, `rri_evp`) have two backends.

A `-DRRI_BACKEND=OMP|OPENCL` CMake option picks which one `main.c` calls
into at runtime (a small vtable/function-pointer indirection in
`backend.h`, resolved once at startup based on the build flag or a
`--device=cpu|gpu` CLI flag).

## 7. Validation plan

Do not trust a numerical port on vibes. In order:

1. **Unit-level**: for each hot kernel, generate the same tiny synthetic
   domain (5x5 / 10x10 flat + simple-slope grids — reuse the fixtures
   RRIpy's `tests/` ends up with) and compare kernel output against a
   hand-computed or Fortran-run reference value, tolerance ~1e-9 relative
   (double precision, should be near bit-exact for a straight port with
   no reordering of floating-point sums; note OpenMP reductions and
   OpenCL work-group orderings can perturb the last few ULPs if you ever
   add tree-reductions — the current per-cell-independent kernels have no
   such reduction inside them, so exact reproducibility across CPU/OMP/CL
   is actually achievable and worth asserting in tests).
2. **Mass balance**: run the tiny synthetic domain end-to-end for a
   handful of timesteps and assert total storage (rainfall in − ET/infil
   losses − outlet flux = Δstorage) closes to a tight tolerance, same
   check the RRIpy pytest smoke test will already do — literally diff the
   two ports' storage.dat-equivalent output.
3. **Cross-backend agreement**: run the same domain through `rri_cpu`
   (serial), `rri_cpu` (OpenMP), and `rri_cl` (OpenCL) and assert they
   agree to the tolerance in (1). This is the actual point of writing
   `kernels.h` once — divergence here means the shared body isn't
   actually shared, or a backend has a race/synchronization bug.
4. **Reference domain**: once (1)-(3) pass, run against one of the real
   example domains that ship with the Fortran repos (check
   `$HOME/dev/RRI_1.4.2.7_Linux` for a `sample`/`test` data dir — if none
   is present, ask the user, don't fabricate a "realistic" domain) and
   compare hydrographs against the Fortran binary's own output.

## 8. Milestones (suggested order)

1. `rri_sub.c` + `rri_section.c` + `globals.h`: GIS I/O, idx setting,
   storage_calc, section geometry. No physics yet — get data structures
   and index-mapping right first, unit-test them alone.
2. `rri_read.c` + `main.c` STEP 0/1/2 skeleton, reads config + grids,
   builds `rri_state`, no time loop yet.
3. `rri_riv.c` (OpenMP only): `funcr`/`qr_calc`/`hq_riv` + the RK45
   adaptive time loop for river routing alone (slope held static/zero).
   This is the first point at which end-to-end validation (§7.2) becomes
   possible on a river-only toy case.
4. `rri_slope.c`, `rri_rivslo.c` (OpenMP): add slope routing and
   river-slope exchange, full coupled RK45 loop.
5. `rri_gw.c`, `rri_infilt.c`, `rri_evp.c` (OpenMP): groundwater,
   infiltration, evaporation — these feed into the RK loop's source terms.
6. `rri_dam.c`, `rri_div.c`, `rri_bound.c` (host-side, no GPU needed):
   wire in dam/diversion/boundary handling.
7. `rri_tecout.c`: output writers, hydrograph files, matching Fortran's
   ascii/binary output format so downstream tooling (GRASS import,
   plotting) doesn't need to change.
8. **DONE.** OpenCL backend: kernel bodies already lived in `kernels.h`
   (written that way from the start, see its file-level comment);
   `.cl` wrappers added for `rri_riv`/`rri_slope`/`rri_gw`/`rri_infilt`
   (`rri_evp` doesn't exist -- evapotranspiration isn't implemented in
   this port at all, CPU or GPU, see README.md), validated per §7.3
   (cross-backend agreement, near-bit-exact) both locally against PoCL
   and on a real GPU (AMD Polaris10 via Mesa Clover). One deviation from
   §6's build-system design: kernel source is concatenated from
   `kernels.h` + `cl/rri_kernels.cl` at RUNTIME (paths baked in as
   compile definitions) rather than embedded into the binary at build
   time (`bin2c`-style) -- simpler to implement correctly under time
   pressure and works fine for validation, but means the source tree
   must travel with the binary (see README.md's remote-GPU procedure,
   which rsyncs the whole tree); embedding is still open if a
   binary-only deployment is ever needed. `-DRRI_BACKEND=OMP|OPENCL`
   build-time selection was also not implemented; instead `main.c` links
   both backends unconditionally and picks at RUNTIME via a `--gpu` CLI
   flag, which is more convenient for the actual use case here (compare
   CPU and GPU output from one binary on one dataset) though it does mean
   an OpenCL toolchain is now a hard build dependency, not optional.
9. `rri_tsas.c`: particle tracking, last, after the core loop is trusted.
10. **DONE (first pass).** Performance: profiled by direct code
    inspection (confirmed the OpenCL backend was re-uploading every
    kernel argument, including unchanging topology, on every RK45
    stage call — the dominant cost, not kernel compute time), then
    fixed via a persistent-buffer redesign (`src/rri_opencl.c`):
    topology/parameters uploaded once per cellset, only the genuinely
    time-varying state transferred per stage. Measured result on
    solo30s (360h, real remote GPU): ~115s → ~87s, correctness
    unchanged (re-validated against Fortran at the same tolerance). CPU
    32-core OpenMP still wins on this problem size (~35s) — see
    README.md's "OpenCL backend" section for the full before/after and
    the two remaining paths considered (flux-scatter moved to a GPU
    kernel; a larger domain, untested so far, to see if a CPU/GPU
    crossover exists). Neither was attempted in this pass; both are the
    natural next steps if GPU throughput becomes the actual goal rather
    than just correctness.

## 9. Open questions for the user before implementation starts

- A real sample dataset already exists: `$HOME/dev/RRI_1.4.2.7_Linux/solo30s/`
  (topo/, riv/, rain/, hs/, out/ — this is also the dataset
  `RRI_Input.py`'s hardcoded `datadir` points at, under a different path).
  Use this for §7.4 end-to-end validation once the core loop is trusted —
  confirm with the user whether this is the intended validation domain or
  just a leftover example before relying on its outputs as ground truth.
- **Answered**: target OpenCL platform turned out to be an AMD Polaris10
  GPU via Mesa's Clover platform, OpenCL 1.1 (not 1.2/3.0 -- this
  mattered concretely: OpenCL C 1.1 rejects `static` on a function at
  kernel-source scope, which `kernels.h`'s `RRI_INLINE` macro had to
  branch on, and requires the `cl_khr_fp64` pragma explicitly rather
  than accepting `double` unconditionally). No workgroup-size tuning was
  done (default/NULL local work size throughout) -- fine for
  correctness validation, a candidate for the throughput pass in
  milestone 10 above.
- Priority: is OpenMP-only (CPU, milestone 7) a useful deliverable on its
  own before investing in the OpenCL backend, or is GPU execution the
  actual point and CPU is just the correctness reference?

## 10. GPU validation host (added post-planning)

Once the C/OpenMP/OpenCL port is implemented and passing cross-backend
validation against PoCL (CPU-only OpenCL ICD, used for correctness only —
see §9), real GPU testing will happen on `yann@10.42.0.89`, made available
by the user at that point. Do not attempt to reach that host during the
initial implementation/PoCL-validation phase — this is a note for the
next phase, not a current dependency.
