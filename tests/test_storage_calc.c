/* storage_calc: slope storage should equal sum(hs)*area over in-domain
 * cells; river storage must be INCLUDED whenever riv_thresh>=0 (this was
 * a real bug found in the Python port: riv_thresh==0 was checked instead
 * of >=0, silently dropping river storage from the mass balance whenever
 * riv_thresh was nonzero -- see RRIpy's README "known gaps" history).
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
        g.land[p] = 1; g.domain[p] = 1;
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
    rri_landuse lu; memset(&lu, 0, sizeof(lu));
    lu.n = 1;
    double one = 1.0, zero = 0.0;
    int dif1 = 1;
    lu.dif = &dif1;
    lu.ns_slope = &one; lu.soildepth = &one; lu.gammaa = &one;
    lu.ksv = &zero; lu.faif = &zero; lu.infilt_limit = &zero;
    lu.ka = &zero; lu.gammam = &zero; lu.beta = &one;
    lu.da = &zero; lu.dm = &zero;
    lu.ksg = &zero; lu.gammag = &one; lu.kg0 = &zero; lu.fpg = &one; lu.rgl = &zero;
    rri_slo_cellset sc;
    assert(rri_slo_idx_setting(&g, &lu, 1, &sc) == 0);

    double *hs = calloc(ny * nx, sizeof(double));
    double *hr = calloc(ny * nx, sizeof(double));
    double *hg = calloc(ny * nx, sizeof(double));
    double *gampt_ff = calloc(ny * nx, sizeof(double));
    for (int p = 0; p < ny * nx; p++) hs[p] = 0.1;
    for (int i = 0; i < ny; i++) hr[i * nx + 1] = 0.3;

    rri_storage s0 = rri_storage_calc(&g, hs, hr, hg, gampt_ff, &sc, &rc, /*riv_thresh=*/-1);
    assert(fabs(s0.sr) < 1e-12); /* riv_thresh<0: river storage excluded */

    rri_storage s1 = rri_storage_calc(&g, hs, hr, hg, gampt_ff, &sc, &rc, /*riv_thresh=*/20);
    double expect_ss = 0.1 * g.area * (ny * nx);
    double expect_sr = 0.0;
    for (int i = 0; i < ny; i++) expect_sr += rri_hr2vr(0.3, g.area, g.width[i * nx + 1] * g.len_riv[i * nx + 1] / g.area);
    assert(fabs(s1.ss - expect_ss) < 1e-6);
    assert(fabs(s1.sr - expect_sr) < 1e-6);
    assert(s1.sr > 0.0); /* the actual regression: must be included when riv_thresh>=0, not just ==0 */

    printf("test_storage_calc: OK (ss=%.3f sr=%.3f)\n", s1.ss, s1.sr);
    return 0;
}
