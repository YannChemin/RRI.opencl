/* qr_calc / qs_calc / qg_calc must return zero flux when there is no head
 * gradient between a cell and its downstream neighbor (flat water table on
 * a flat bed). Uses the same tiny 3x3 synthetic domain as test_idx_setting,
 * but with a FLAT bed (zb=0 everywhere) instead of the sloped one there.
 */
#include "rri/rri.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    int ny = 3, nx = 3;
    rri_grid g; memset(&g, 0, sizeof(g));
    g.ny = ny; g.nx = nx;
    g.dx = 30.0; g.dy = 30.0; g.area = g.dx * g.dy; g.length = 30.0;
    g.zb = calloc(ny * nx, sizeof(double));
    g.zb_riv = calloc(ny * nx, sizeof(double));
    g.acc = calloc(ny * nx, sizeof(double));
    g.dir = calloc(ny * nx, sizeof(int));
    g.domain = calloc(ny * nx, sizeof(int));
    g.riv = calloc(ny * nx, sizeof(int));
    g.land = calloc(ny * nx, sizeof(int));
    g.width = calloc(ny * nx, sizeof(double));
    g.depth = calloc(ny * nx, sizeof(double));
    g.height = calloc(ny * nx, sizeof(double));
    g.len_riv = calloc(ny * nx, sizeof(double));
    g.area_ratio = calloc(ny * nx, sizeof(double));

    for (int i = 0; i < ny; i++) for (int j = 0; j < nx; j++) {
        int p = i * nx + j;
        g.land[p] = 1;
        g.domain[p] = 1;
        g.dir[p] = (i == ny - 1) ? 0 : 4;
    }
    g.domain[(ny - 1) * nx + 1] = 2;
    for (int i = 0; i < ny; i++) {
        int p = i * nx + 1;
        g.riv[p] = 1;
        g.width[p] = 5.0; g.depth[p] = 1.0; g.len_riv[p] = g.length;
        g.area_ratio[p] = g.width[p] * g.len_riv[p] / g.area;
    }

    rri_riv_cellset rc;
    assert(rri_riv_idx_setting(&g, &rc) == 0);

    double *hr_idx = calloc(rc.count, sizeof(double));
    for (int k = 0; k < rc.count; k++) hr_idx[k] = 0.5; /* uniform depth, flat bed -> zero gradient everywhere */
    double *qr_idx = calloc(rc.count, sizeof(double));
    rri_qr_calc(&rc, hr_idx, 0.03, qr_idx);
    for (int k = 0; k < rc.count; k++) {
        if (rc.domain[k] == 2) continue; /* outlet cell: qr forced 0 by rri_qr_calc, not part of this check */
        if (rc.domain[rc.down[k]] == 2) continue; /* cell draining directly into the outlet legitimately
            gets a nonzero head gradient even on a flat bed (qr_calc's outlet special case uses
            zb_p+hr_p-zb_n, i.e. the water depth itself drives outflow, not just bed slope) */
        /* dh==0 branch in qr_calc takes the "dh>=0" path with hw=hr_p and dh=0,
         * so hq_riv is called with dh=0 -> a=0 -> q=0. */
        assert(fabs(qr_idx[k]) < 1e-12);
    }

    rri_landuse lu; memset(&lu, 0, sizeof(lu));
    lu.n = 1;
    double one = 1.0, zero = 0.0;
    int dif1 = 1;
    lu.dif = &dif1;
    lu.ns_slope = &one; lu.soildepth = &one; lu.gammaa = &one;
    lu.ksv = &zero; lu.faif = &zero; lu.infilt_limit = &zero;
    lu.ka = &zero; lu.gammam = &zero; lu.beta = &one;
    lu.da = &zero; lu.dm = &zero;
    lu.ksg = &one; lu.gammag = &one; lu.kg0 = &one; lu.fpg = &one; lu.rgl = &zero;

    rri_slo_cellset sc;
    assert(rri_slo_idx_setting(&g, &lu, 1, &sc) == 0);

    double *hs_idx = calloc(sc.count, sizeof(double));
    double *hg_idx = calloc(sc.count, sizeof(double));
    for (int k = 0; k < sc.count; k++) { hs_idx[k] = 0.2; hg_idx[k] = 0.1; }
    double *qs_buf[RRI_LMAX8], *qg_buf[RRI_LMAX8];
    for (int l = 0; l < RRI_LMAX8; l++) { qs_buf[l] = calloc(sc.count, sizeof(double)); qg_buf[l] = calloc(sc.count, sizeof(double)); }

    rri_qs_calc(&sc, hs_idx, g.area, qs_buf);
    rri_qg_calc(&sc, hg_idx, g.area, qg_buf);
    for (int k = 0; k < sc.count; k++) {
        for (int l = 0; l < sc.lmax; l++) {
            assert(fabs(qs_buf[l][k]) < 1e-12);
            assert(fabs(qg_buf[l][k]) < 1e-12);
        }
    }

    printf("test_kernels_zero_gradient: OK\n");
    return 0;
}
