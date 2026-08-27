# RRI.opencl

C11 port of the RRI rainfall-runoff-inundation model, following the plan in
[`PLAN.md`](PLAN.md). Status: **core solver implemented and validated
against the compiled Fortran reference on the real solo30s dataset over
its full 360-hour run (<0.4% relative error throughout, including the
flood peak); OpenCL backend not yet started.**

## Physics overview

RRI simulates rainfall running off a watershed to its outlet, coupling
four processes, each advanced together every timestep:

- **River routing**: diffusive-wave flow along a river network (a tree:
  each river cell has exactly one downstream neighbor), rectangular
  channel geometry, Manning's equation.
- **Hillslope routing**: diffusive-wave (up to 8 neighbor directions) or
  kinematic-wave (1 direction, following D8 flow direction) overland +
  shallow subsurface flow, receiving rainfall directly.
- **Groundwater**: a shallow bedrock aquifer exchanging water laterally
  between cells and vertically with the surface (recharge when the
  surface has standing water, exfiltration when the aquifer overflows).
- **Green-Ampt infiltration**: surface water infiltrating into the soil
  column at a rate that decays as cumulative infiltration grows.

River and hillslope cells additionally exchange water directly wherever
they're collocated (river<->slope exchange -- a river cell also has an
hillslope depth at the same location, and water moves between the two
via weir-flow-like formulas whenever the river overtops its banks or a
hillslope cell drains into an under-full channel).

All of this is advanced through time by an **adaptive Runge-Kutta-
Fehlberg (Cash-Karp) 4(5) integrator**, applied independently to the
river, hillslope, and groundwater state each outer timestep -- each gets
its own accept/reject step-size sequence within the shared outer `dt`.
See `include/rri/kernels.h` for the innermost discharge formulas and
`src/main.c`'s file-level comment for the full coupling order and RK45
control-flow structure.

## Data model: sparse index representation

The model grid (elevation, land cover, river geometry) is a full
raster, but the actual state the RK45 integrator advances is stored
compressed into 1D arrays covering only the *active* cells (`rri_riv_cellset`
for river cells, `rri_slo_cellset` for every in-domain cell), built once
at startup by `rri_riv_idx_setting`/`rri_slo_idx_setting`. Each active
cell `k` carries a precomputed neighbor lookup -- `down[k]` (or
`down[l][k]` for hillslope cells' up to 4 direction slots), `dis[k]`
(distance), `len[k]` (shared-edge contour length) -- instead of
re-deriving "what's adjacent to cell k" from (i,j) arithmetic every
timestep.

This is a CSR-like sparse graph, and it's *why the per-cell discharge
kernels parallelize cleanly*: each kernel invocation for cell k reads
only k's own state and its precomputed neighbors' state from the
previous timestep, and writes only k's own output -- no cell-to-cell
dependency within one kernel call. That's what makes `rri_qr_calc`/
`rri_qs_calc`/`rri_qg_calc` safe under OpenMP today (`#pragma omp
parallel for` over the cellset) and is the exact seam the OpenCL work
will build on next (an `NDRange` over the same index). The parts of the
solver that are NOT structured this way -- the flux-scatter step that
sums each cell's outflow into its downstream neighbor's inflow (a
shared-destination write across loop iterations) and the RK45 accept/
reject control flow itself -- stay serial/host-side by design; see
`include/rri/rri.h`'s file-level comment for the full rationale.

## Build & test

```
cmake -B build && cmake --build build
ctest --test-dir build --output-on-failure
```

Requires a C11 compiler and OpenMP (auto-detected by CMake; the build
still succeeds without it, just without the `#pragma omp` parallelism).
An OpenCL SDK is detected if present but unused (see "What's NOT
implemented" below).

## What's implemented (OpenMP-parallel CPU, `rri_cpu`)

- ESRI ASCII grid I/O (whitespace- and comma-separated variants both
  accepted), `RRI_Input.txt` config parsing (full field order, matches
  `RRI_Read.f90`).
- River + slope index setting (grid <-> compressed 1D cellset), rectangular
  channel geometry (`hr2vr`/`vr2hr`), `storage_calc`. Two DISTINCT bed
  elevation arrays, `grid.zb` (slope bed = dem - soildepth[land]) and
  `grid.zb_riv` (river channel bed = dem - depth), matching RRI.f90 lines
  ~223-224 exactly -- see "Root-caused bug" below for why this distinction
  matters and what breaks without it.
- Diffusive-wave river routing (`funcr`/`qr_calc`) and slope routing
  (`funcs`/`qs_calc`), coupled via adaptive Cash-Karp RK45 exactly as in
  `RRI.f90`'s main time loop (verbatim RK coefficients).
- River <-> slope exchange (`funcrs`), groundwater lateral flow +
  recharge/exfiltration (`funcg`, `gw_recharge`, `gw_lose`, `gw_exfilt`),
  Green-Ampt infiltration.
- `hydro.txt` / `hydro_hr.txt` (outlet discharge time series) and
  `storage.dat` (mass balance) output -- the two outputs needed to
  validate against the Fortran binary.

## What's NOT implemented (see PLAN.md; all off in the validated solo30s
config, so none of this blocks the validation below)

- Dam, diversion, boundary conditions (water-level/discharge), custom
  cross-sections (`sec_map`), evapotranspiration, initial-condition files.
- `rivfile_switch>=1` (river geometry read from files instead of the
  parametric `width_param_*`/`depth_param_*` formula) -- `main` exits with
  an error if the config requests any of the above rather than silently
  ignoring them.
- Periodic full-grid output (`hs_*`/`hr_*`/`qr_*`/... snapshot files) --
  only the two time-series outputs above are written.
- TSAS particle tracking.
- **OpenCL backend.** `kernels.h` deliberately isolates the per-cell math
  (rectangular `hq_riv`, slope `hq`, `h2lev`, groundwater `hg_calc`) as
  pure scalar functions with no pointers/structs, specifically so they
  compile unchanged as OpenCL C kernel bodies -- but no `.cl` files or
  host-side dispatch exist yet. This is PLAN.md milestone 8, the natural
  next step now that the CPU reference is validated.

## Validation

Built with `cmake -B build && cmake --build build`, then `ctest` runs 6
unit tests (GIS I/O round-trip incl. header-mismatch rejection, geodesic
distance, rectangular-section `hr2vr`/`vr2hr` inverses, index-setting
consistency on a synthetic domain, zero-head-gradient implies zero flux
for the river/slope/groundwater kernels, `storage_calc` -- including a
regression test for the exact `riv_thresh>=0` vs `riv_thresh==0` bug the
Python port hit).

Real-dataset validation against the compiled Fortran binary
(`$HOME/dev/RRI_1.4.2.7_Linux`, `make` -> `0_rri_1_4_2_7_Linux`) on
`solo30s` (336x204 cells), same procedure as RRIpy's:

```
cp -r $HOME/dev/RRI_1.4.2.7_Linux/solo30s/* <dir>
# shorten RRI_Input.txt's lasth for a fast comparison, e.g. sed -i 's/^360 /26 /'
./build/rri_cpu <dir>/
```

Results (26-hour window, includes the dataset's actual rainfall onset at
t=24h, so this exercises the full coupled solver under real forcing, not
just a dry warm-up):

- `storage.dat`: max relative error **0.19%** (worst column is `sout`,
  cumulative outlet loss -- the other columns, including total storage,
  are tighter); the water-balance residual column agrees to 5-6
  significant figures in absolute terms with the Fortran run's own
  residual (both ~5.5e-4, i.e. this is the same floating-point-accumulation
  noise in both implementations, not a divergence).
- `hydro.txt` (outlet discharge): matches to within **0.3%** at every
  hourly sample through the rain event (e.g. final sample: 0.27736 vs
  Fortran's 0.27787 m^3/s).
- Wall-clock: ~1s vs the Fortran binary's ~6.5s for this window (not a
  rigorous benchmark -- the Fortran binary also writes more output files
  -- but a reasonable sanity check that the OpenMP core isn't pathologically
  slow).

A shorter 1-hour dry-warmup comparison (no rain yet) matches to
3.1e-7 relative.

### Porting gotchas

Fast-reference list for anyone porting more of RRI.f90 (or auditing this
one) -- each bit us once and is now handled, but the pattern is worth
knowing before it bites again in unported code:

- **`zb` vs `zb_riv` are two DISTINCT bed-elevation arrays, never one.**
  `zb` = dem - soildepth[land] (every cell, used by hillslope/groundwater
  routing); `zb_riv` = dem - channel depth (river cells only, used by
  river routing). Conflating them (e.g. feeding the raw DEM to both, or
  building the river cellset from `grid::zb`) does NOT crash and is
  invisible at small water depths -- it produces a slowly-growing,
  one-directional mass-balance drift that only shows up on a multi-day
  run. See "Root-caused bug" below for the full story; see `rri.h`:
  `rri_grid::zb`/`zb_riv`'s doc for where each is used.
- **The RK45 accept/reject error norm is the SIGNED maximum
  (`maxval(err)/eps` in Fortran), not `maxval(fabs(err))`.** A large
  negative per-cell error must NOT trigger a step shrink. Using `fabs()`
  there is a plausible-looking mistake that still produces a
  numerically-stable, physically-plausible-looking trajectory -- it just
  doesn't match the Fortran reference's exact step sequence. See
  `rri.h`: `rri_rk_coeffs`'s doc and the inline comment at every
  `errmax = -DBL_MAX` in `main.c`.
- **A short validation window is not sufficient evidence of a correct
  port.** Both bugs above were invisible at 1h and 26h of simulated time
  and were only found by running the dataset's full 360-hour duration.
  If you change anything touching bed elevation, channel geometry, or
  the RK45 control flow, re-run the full-length comparison before
  trusting it, not just a short smoke test.
- **Grid files in this dataset family are inconsistently
  comma-separated** (`acc_mod.txt`/`dir_mod.txt`/`adem.txt` use commas;
  `acc.txt`/`dir.txt`/`dem.txt` use whitespace) -- `rri_read_gis_real`/
  `rri_read_gis_int` tokenize on both rather than assuming one convention.
- **Sign convention**: every discharge array in this codebase (`qr_idx`,
  `qs_idx[l]`, `qg_idx[l]`) is positive = outflow from the cell it's
  indexed by, negative = inflow (computed using the neighbor's state,
  stored at this cell's index with a negated sign) -- never by swapping
  which cell is treated as "the source". This lets the flux-scatter step
  treat every cell uniformly: `sum[k] += q[k]; sum[down[k]] -= q[k]` is
  correct regardless of physical flow direction. See `rri_riv.c`'s
  file-level comment for the full explanation.

### Root-caused bug: missing river-channel bed incision (fixed)

An earlier pass of this port validated only short windows (1h dry, 26h
including rain onset, both <0.4% error) and initially concluded the
port was solid. **Running the full 360-hour comparison exposed a much
larger divergence: storage.dat relative error up to 57%, and hydro.txt
discharge at the flood peak (t=180h) off by ~53% (Fortran 2114 vs C 1914
m^3/s).** The coordinator correctly pushed back on attributing this to
"chaotic sensitivity in an independently re-implemented adaptive
stepper" without first ruling out an actual bug -- identical RK45
coefficients given identical physics and identical inputs should not
chaotically diverge; if it does, something is inconsistent between the
implementations.

Bisection: plotting relative storage error vs. simulated hour showed it
was ~0 through hour 24, then grew smoothly and monotonically from hour
~36 onward, with the C port always retaining *more* water on the slope
(`ss`) than the Fortran run, and correspondingly less in the river
(`sr`). A monotonic, one-directional drift (not oscillating) pointed at
a systematic capacity/gradient bug rather than genuine chaotic RK
step-sequence divergence (which would be expected to look more erratic).
Comparing `RRI.f90` line-by-line against this port's grid setup found it:

**Fortran maintains two separate bed-elevation arrays** -- `zb` (slope
bed = dem - soildepth[land], RRI.f90 ~line 223) and `zb_riv` (river
channel bed = dem - depth, ~line 224, river cells only). RRIpy correctly
replicates this split. This C port's `main.c` had instead used the raw
DEM directly as the single bed elevation for *both* river and slope
routing, with no channel incision at all. That gives the river channel
almost no elevation/capacity advantage over the surrounding slope, so
under real rainfall loading water preferentially pools on the slope
instead of draining into the (artificially shallow) channel -- consistent
with the observed pattern, small at first (depths too shallow for the
missing few-meter offset to matter) and compounding as flow builds.

Also found and fixed in the same pass, independently confirmed correct
per `RRI.f90` but worth noting since it was a real discrepancy from the
naive translation: the RK45 accept/reject error norm. **Fortran uses
`errmax = maxval(hr_err) / eps` -- the plain (signed) maximum, not
`maxval(abs(hr_err))`.** This port's `main.c` took `fabs()` of each
per-cell error before maxing, which shrinks the timestep on large
*negative* errors that Fortran's unsigned max ignores. This alone did not
meaningfully move the full-run numbers (tested in isolation: still ~57%
divergence), so the bed-elevation fix was the dominant cause -- but it's
a genuine formula mismatch and is fixed regardless.

**Full 360-hour comparison, before -> after both fixes:**

| metric | before | after |
|---|---|---|
| storage.dat max relative error | 57% | **0.069%** |
| hydro.txt max relative error | 53% (at the flood peak) | **0.37%** |
| hydro.txt mean relative error | 10% | **0.16%** |
| final total storage (Fortran vs C) | 2.267e9 vs 2.729e9 | 2.26665e9 vs 2.26617e9 |

This is now within the same tight tolerance the short-window checks
already showed, sustained across the entire run including the flood
peak -- the divergence is root-caused and fixed, not just reduced. All 6
unit tests still pass after both fixes (`tests/test_idx_setting.c`,
`test_kernels_zero_gradient.c`, `test_storage_calc.c` needed a
`zb_riv` field added to their synthetic-domain setup to match the new
`rri_grid` struct).

**Lesson for future validation passes on this codebase:** short-window
comparisons are a fast sanity check, not a substitute for at least one
full-length run before declaring a port "validated" -- this bug was
completely invisible at 1h and 26h and would have shipped undetected.

## Repo layout

Deviates slightly from `PLAN.md`'s exact per-`.f90`-file split for
pragmatism (one `rri_setup.c` covers index setting + section geometry +
storage_calc rather than three separate files) -- module boundaries are
still 1:1 auditable against the Fortran source via the function-level
comments in each file.

```
include/rri/rri.h       All public types + function declarations
include/rri/kernels.h   Shared per-cell math (river/slope/gw hq relations,
                         h2lev) -- written once for eventual OpenMP+OpenCL sharing
src/rri_io.c            GIS grid I/O, RRI_Input.txt config parser, hubeny_sub
src/rri_setup.c         idx setting, ij2idx/idx2ij, rectangular section, storage_calc
src/rri_rk.c            Cash-Karp RK45 coefficients (verbatim from RRI.f90)
src/rri_riv.c           funcr / qr_calc (OpenMP)
src/rri_slope.c         funcs / qs_calc (OpenMP)
src/rri_gw.c            funcg / qg_calc / gw_recharge / gw_lose / gw_exfilt (OpenMP)
src/rri_infilt.c        Green-Ampt infiltration (OpenMP)
src/rri_rivslo.c        funcrs (river<->slope exchange, host-side)
src/main.c              STEP 0-2 setup + main adaptive-RK45 time loop
tests/                  6 unit tests (CTest)
```

## Next steps (PLAN.md milestone order)

1. OpenCL backend (milestone 8): `.cl` kernels for `qr_calc`/`qs_calc`
   sharing `kernels.h`'s math bodies, host dispatch, cross-backend
   agreement test (CPU serial vs OpenMP vs OpenCL on the synthetic
   domain) per PLAN.md section 7.3. Now that the CPU core is validated
   end-to-end on the full 360-hour solo30s run (see above), this is
   unblocked.
2. Dam, diversion, boundary conditions, custom cross-sections if the
   project needs domains that use them.
3. Consider porting the (currently omitted) levee-height DEM adjustment
   in main.c's grid setup (`zs += height` before deriving zb/zb_riv when
   `height_param>0`) before running any dataset that sets it nonzero --
   solo30s uses 0 so this didn't affect validation, but it's a known gap.
