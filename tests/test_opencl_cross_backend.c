/**
 * @file test_opencl_cross_backend.c
 * @brief Cross-backend agreement (PLAN.md section 7.3): the OpenCL
 * kernels in cl/rri_kernels.cl must produce the same numbers as their
 * OpenMP counterparts in src/rri_riv.c/rri_slope.c/rri_gw.c/rri_infilt.c,
 * since both compile the exact same math bodies from kernels.h. Run
 * against whatever OpenCL device rri_cl_backend_init(prefer_gpu=0)
 * selects locally (PoCL, a CPU-only OpenCL implementation, in this
 * project's development environment) -- see README.md's OpenCL section
 * for the separate procedure used to validate against a real GPU
 * (Mesa Clover / AMD Polaris10) on a remote host, which this test does
 * NOT reach (no network access assumed for the local test suite).
 *
 * Uses a 5x5 synthetic domain with VARYING (not uniform/degenerate)
 * elevation and water depth, so agreement here means the kernels
 * actually compute the same nontrivial per-cell results, not just that
 * both return zero on a trivial input.
 */
#include "rri/rri.h"
#include "rri/opencl.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void build_domain(rri_grid *g, int ny, int nx)
{
    memset(g, 0, sizeof(*g));
    g->ny = ny; g->nx = nx;
    g->dx = 25.0; g->dy = 25.0; g->area = g->dx * g->dy; g->length = 25.0;
    g->zb = calloc(ny * nx, sizeof(double));
    g->zb_riv = calloc(ny * nx, sizeof(double));
    g->acc = calloc(ny * nx, sizeof(double));
    g->dir = calloc(ny * nx, sizeof(int));
    g->domain = calloc(ny * nx, sizeof(int));
    g->riv = calloc(ny * nx, sizeof(int));
    g->land = calloc(ny * nx, sizeof(int));
    g->width = calloc(ny * nx, sizeof(double));
    g->depth = calloc(ny * nx, sizeof(double));
    g->height = calloc(ny * nx, sizeof(double));
    g->len_riv = calloc(ny * nx, sizeof(double));
    g->area_ratio = calloc(ny * nx, sizeof(double));

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int p = i * nx + j;
            g->land[p] = 1;
            g->domain[p] = 1;
            g->dir[p] = (i == ny - 1) ? 0 : 4; /* straight-down flow, outlet at the bottom row */
            g->zb[p] = (ny - 1 - i) * 1.3 + j * 0.2; /* nontrivial slope, varies by row AND column */
        }
    }
    g->domain[(ny - 1) * nx + 2] = 2;
    for (int i = 0; i < ny; i++) {
        int p = i * nx + 2; /* river down the middle column */
        g->riv[p] = 1;
        g->width[p] = 4.0 + 0.3 * i;
        g->depth[p] = 1.0;
        g->len_riv[p] = g->length;
        g->area_ratio[p] = g->width[p] * g->len_riv[p] / g->area;
        g->zb_riv[p] = g->zb[p] - g->depth[p];
    }
}

static void make_landuse(rri_landuse *lu)
{
    memset(lu, 0, sizeof(*lu));
    lu->n = 1;
    static double one = 1.0, half = 0.5, small = 0.05, ks = 5.0e-6, ks0 = 5.0e-4, dif1_val;
    static int dif1 = 1;
    lu->dif = &dif1;
    lu->ns_slope = &half; lu->soildepth = &one; lu->gammaa = &half;
    lu->ksv = &small; lu->faif = &small; lu->infilt_limit = &half;
    lu->ka = &small; lu->gammam = &small; lu->beta = &one;
    lu->da = &half; lu->dm = &small;
    lu->ksg = &ks; lu->gammag = &half; lu->kg0 = &ks0; lu->fpg = &small; lu->rgl = &small;
    (void)dif1_val;
}

static double max_abs_diff(const double *a, const double *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) { double d = fabs(a[i] - b[i]); if (d > m) m = d; }
    return m;
}

int main(void)
{
    int ny = 5, nx = 5;
    rri_grid g; build_domain(&g, ny, nx);
    rri_landuse lu; make_landuse(&lu);

    rri_riv_cellset rc;
    if (rri_riv_idx_setting(&g, &rc) != 0) { fprintf(stderr, "riv idx setting failed\n"); return 1; }
    rri_slo_cellset sc;
    if (rri_slo_idx_setting(&g, &lu, 1, &sc) != 0) { fprintf(stderr, "slo idx setting failed\n"); return 1; }

    rri_cl_backend *b = rri_cl_backend_init(/*prefer_gpu=*/0);
    if (!b) { fprintf(stderr, "test_opencl_cross_backend: no OpenCL device available -- skipping\n"); return 0; }
    printf("OpenCL device: %s\n", rri_cl_backend_device_name(b));

    /* ---- river: qr_calc, CPU vs OpenCL ---- */
    double *hr_idx = malloc(sizeof(double) * rc.count);
    for (int k = 0; k < rc.count; k++) hr_idx[k] = 0.1 + 0.05 * k;
    double *qr_cpu = malloc(sizeof(double) * rc.count);
    double *qr_cl = malloc(sizeof(double) * rc.count);
    rri_qr_calc(&rc, hr_idx, 0.03, qr_cpu);
    rri_cl_qr_calc(b, &rc, hr_idx, 0.03, qr_cl);
    double dqr = max_abs_diff(qr_cpu, qr_cl, rc.count);
    printf("qr_calc max abs diff (CPU vs OpenCL): %.3e\n", dqr);
    assert(dqr < 1e-9);

    /* ---- slope: qs_calc, CPU vs OpenCL ---- */
    double *hs_idx = malloc(sizeof(double) * sc.count);
    for (int k = 0; k < sc.count; k++) hs_idx[k] = 0.05 + 0.01 * (k % 7);
    double *qs_cpu[RRI_LMAX8], *qs_cl[RRI_LMAX8];
    for (int l = 0; l < RRI_LMAX8; l++) { qs_cpu[l] = calloc(sc.count, sizeof(double)); qs_cl[l] = calloc(sc.count, sizeof(double)); }
    rri_qs_calc(&sc, hs_idx, g.area, qs_cpu);
    rri_cl_qs_calc(b, &sc, hs_idx, g.area, qs_cl);
    double dqs = 0.0;
    for (int l = 0; l < RRI_LMAX8; l++) { double d = max_abs_diff(qs_cpu[l], qs_cl[l], sc.count); if (d > dqs) dqs = d; }
    printf("qs_calc max abs diff (CPU vs OpenCL): %.3e\n", dqs);
    assert(dqs < 1e-9);

    /* ---- groundwater: qg_calc, CPU vs OpenCL ---- */
    double *hg_idx = malloc(sizeof(double) * sc.count);
    for (int k = 0; k < sc.count; k++) hg_idx[k] = -0.02 + 0.005 * (k % 9);
    double *qg_cpu[RRI_LMAX8], *qg_cl[RRI_LMAX8];
    for (int l = 0; l < RRI_LMAX8; l++) { qg_cpu[l] = calloc(sc.count, sizeof(double)); qg_cl[l] = calloc(sc.count, sizeof(double)); }
    rri_qg_calc(&sc, hg_idx, g.area, qg_cpu);
    rri_cl_qg_calc(b, &sc, hg_idx, g.area, qg_cl);
    double dqg = 0.0;
    for (int l = 0; l < RRI_LMAX8; l++) { double d = max_abs_diff(qg_cpu[l], qg_cl[l], sc.count); if (d > dqg) dqg = d; }
    printf("qg_calc max abs diff (CPU vs OpenCL): %.3e\n", dqg);
    assert(dqg < 1e-9);

    /* ---- infiltration: rri_infilt, CPU vs OpenCL (in-place, so run on two separate copies) ---- */
    double *gff_cpu = calloc(sc.count, sizeof(double)), *gff_cl = calloc(sc.count, sizeof(double));
    double *gf_cpu = calloc(sc.count, sizeof(double)), *gf_cl = calloc(sc.count, sizeof(double));
    double *hsi_cpu = malloc(sizeof(double) * sc.count), *hsi_cl = malloc(sizeof(double) * sc.count);
    for (int k = 0; k < sc.count; k++) { hsi_cpu[k] = hsi_cl[k] = 0.02 + 0.001 * k; }
    rri_infilt(&sc, 600.0, hsi_cpu, gff_cpu, gf_cpu);
    rri_cl_infilt(b, &sc, 600.0, hsi_cl, gff_cl, gf_cl);
    double dinf = max_abs_diff(hsi_cpu, hsi_cl, sc.count);
    double dff = max_abs_diff(gff_cpu, gff_cl, sc.count);
    printf("infilt max abs diff (hs %.3e, gampt_ff %.3e)\n", dinf, dff);
    assert(dinf < 1e-9);
    assert(dff < 1e-9);

    rri_cl_backend_free(b);
    printf("test_opencl_cross_backend: OK\n");
    return 0;
}
