/**
 * @file opencl.h
 * @brief Host-side OpenCL backend: device selection, program build, and
 * per-kernel dispatch functions with the SAME signatures as their OpenMP
 * counterparts in rri.h (rri_qr_calc / rri_qs_calc / rri_qg_calc /
 * rri_infilt), so a caller can swap backends without touching call sites
 * beyond the function name.
 *
 * This is a "correct first" dispatch design: each `rri_cl_*` call
 * uploads its inputs, runs the kernel, downloads the outputs, and
 * releases the device buffers -- one call = one round trip, no
 * persistent device-resident state across calls. That is intentionally
 * NOT the fastest possible design (a real performance pass would keep
 * `hr_idx`/`hs_idx`/etc. resident on the device across the whole RK45
 * sub-loop and only sync back what's needed each step) -- see PLAN.md
 * milestone 10 ("do not optimize ahead of a profile"). The point of
 * this milestone is validating that the OpenCL kernels (cl/rri_kernels.cl,
 * sharing kernels.h's math bodies with the OpenMP path) produce the
 * same numbers as the CPU backends; a throughput-oriented redesign is
 * future work once that's established.
 *
 * Fails loudly (returns nonzero from rri_cl_init, with a diagnostic on
 * stderr) rather than silently falling back to single precision if the
 * selected device lacks `cl_khr_fp64` -- see rri.h's design-decision
 * doc / PLAN.md section 5 for why (the whole model is written in double
 * precision; a silent float fallback would produce plausible-looking
 * but numerically wrong results).
 */
#ifndef RRI_OPENCL_H
#define RRI_OPENCL_H

#include "rri/rri.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rri_cl_backend rri_cl_backend; /* opaque: real definition (with <CL/cl.h> types) lives in rri_opencl.c */

/**
 * @brief Select an OpenCL device, build the kernel program (from
 * kernels.h + cl/rri_kernels.cl, concatenated at runtime -- see
 * cl/rri_kernels.cl's file-level comment), and create the kernel objects.
 * @param prefer_gpu  1 to prefer a GPU device if one with cl_khr_fp64
 *                    support exists, 0 to force CPU-only device
 *                    selection (used by the PoCL cross-backend
 *                    validation test, which specifically wants the
 *                    CPU-based OpenCL path, not whatever GPU might also
 *                    be present).
 * @return Newly allocated backend on success (caller must
 *         rri_cl_backend_free() it); NULL on failure (device selection,
 *         missing cl_khr_fp64, or a kernel compile error -- all logged
 *         to stderr, including the OpenCL build log on a compile
 *         failure).
 */
rri_cl_backend *rri_cl_backend_init(int prefer_gpu);
/** @brief Release all OpenCL resources held by @p b (kernels, program,
 * queue, context) and free the struct itself. */
void rri_cl_backend_free(rri_cl_backend *b);

/** @brief Human-readable device name/vendor/OpenCL-version string for
 * whatever device rri_cl_backend_init selected, e.g. for logging which
 * backend a validation run actually used. Returned pointer is valid for
 * @p b's lifetime. */
const char *rri_cl_backend_device_name(const rri_cl_backend *b);

/** @brief OpenCL counterpart of rri_qr_calc (src/rri_riv.c); same
 * inputs/outputs/units, dispatched as one OpenCL kernel invocation over
 * @p rc->count work-items instead of an OpenMP loop. */
void rri_cl_qr_calc(rri_cl_backend *b, const rri_riv_cellset *rc, const double *hr_idx,
                     double ns_river, double *qr_idx);
/** @brief OpenCL counterpart of rri_qs_calc (src/rri_slope.c). */
void rri_cl_qs_calc(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hs_idx,
                     double area, double *qs_idx[RRI_LMAX8]);
/** @brief OpenCL counterpart of rri_qg_calc (src/rri_gw.c). */
void rri_cl_qg_calc(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hg_idx,
                     double area, double *qg_idx[RRI_LMAX8]);
/** @brief OpenCL counterpart of rri_infilt (src/rri_infilt.c); updates
 * @p hs_idx / @p gampt_ff_idx / @p gampt_f_idx in place, same as the CPU version. */
void rri_cl_infilt(rri_cl_backend *b, const rri_slo_cellset *sc, double dt,
                    double *hs_idx, double *gampt_ff_idx, double *gampt_f_idx);

/**
 * @name RK45-stage drivers, GPU-backed
 *
 * OpenCL counterparts of rri_funcr/rri_funcs/rri_funcg (src/rri_riv.c,
 * rri_slope.c, rri_gw.c): same signature plus a leading @p b, same
 * semantics (trial-state -> per-cell derivative, one call per Cash-Karp
 * stage) -- used by src/main.c's `--gpu` backend selection to run the
 * SAME adaptive-RK45 time loop with the discharge kernels dispatched to
 * OpenCL instead of OpenMP, without duplicating the ~150-line RK45
 * control-flow logic itself. The host-side flux-scatter step (summing
 * each cell's outflow into its downstream neighbor's inflow -- see
 * rri_funcr/rri_funcs's file-level comments for why this part stays
 * host-side regardless of backend) is plain C in both cases, run after
 * the OpenCL discharge kernel's result is downloaded.
 * @{
 */
void rri_cl_funcr(rri_cl_backend *b, const rri_riv_cellset *rc, const double *vr_idx,
                   double ns_river, double area, double *hr_idx, double *fr_idx,
                   double *qr_idx, double *qr_sum_scratch);
void rri_cl_funcs(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hs_idx,
                   const double *qp_t_idx, double area, double *fs_idx, double *qs_idx[RRI_LMAX8]);
void rri_cl_funcg(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hg_idx,
                   double area, double *fg_idx, double *qg_idx[RRI_LMAX8]);
/** @} */

#ifdef __cplusplus
}
#endif
#endif /* RRI_OPENCL_H */
