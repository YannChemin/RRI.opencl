/* riv_idx_setting / slo_idx_setting on a tiny synthetic domain, built
 * in-memory (no GIS files): consistency checks on cell counts, down[]
 * index ranges, and that the outlet is reachable.
 *
 * Domain: 3x3 grid, single river column (j=1) flowing straight down
 * (dir=4) to an outlet at the bottom (dir=0), 8-direction slope routing.
 */
#include "rri/rri.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    int ny = 3, nx = 3;
    rri_grid g; memset(&g, 0, sizeof(g));
    g.ny = ny; g.nx = nx;
    g.xllcorner = 0; g.yllcorner = 0; g.cellsize = 30.0;
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
        g.zb[p] = (ny - 1 - i) * 1.0; /* slopes down towards i=ny-1 */
        g.zb_riv[p] = g.zb[p]; /* no channel incision in this synthetic domain */
        g.land[p] = 1;
        g.domain[p] = 1;
        g.dir[p] = (i == ny - 1) ? 0 : 4; /* flow straight down, outlet at bottom row */
    }
    g.domain[(ny - 1) * nx + 1] = 2; /* outlet cell, bottom of the river column */
    for (int i = 0; i < ny; i++) {
        int p = i * nx + 1;
        g.riv[p] = 1;
        g.width[p] = 5.0; g.depth[p] = 1.0; g.len_riv[p] = g.length;
        g.area_ratio[p] = g.width[p] * g.len_riv[p] / g.area;
    }

    rri_riv_cellset rc;
    assert(rri_riv_idx_setting(&g, &rc) == 0);
    assert(rc.count == ny); /* one river cell per row */
    for (int k = 0; k < rc.count; k++) {
        assert(rc.down[k] >= 0 && rc.down[k] < rc.count);
    }
    /* the bottom river cell should point to itself (outlet, dir forced to 0) */
    int bottom_k = -1;
    for (int k = 0; k < rc.count; k++) if (rc.idx2i[k] == ny - 1) bottom_k = k;
    assert(bottom_k >= 0);
    assert(rc.down[bottom_k] == bottom_k);

    rri_landuse lu; memset(&lu, 0, sizeof(lu));
    lu.n = 1;
    double one = 1.0, zero = 0.0;
    int dif1 = 1;
    lu.dif = &dif1;
    lu.ns_slope = &one; lu.soildepth = &one; lu.gammaa = &one;
    lu.ksv = &zero; lu.faif = &zero; lu.infilt_limit = &zero;
    lu.ka = &zero; lu.gammam = &zero; lu.beta = &one;
    lu.da = &zero; lu.dm = &zero;
    lu.ksg = &zero; lu.gammag = &zero; lu.kg0 = &zero; lu.fpg = &one; lu.rgl = &zero;

    rri_slo_cellset sc;
    assert(rri_slo_idx_setting(&g, &lu, 1, &sc) == 0);
    assert(sc.count == ny * nx); /* every cell is in-domain */
    assert(sc.lmax == 4);
    for (int k = 0; k < sc.count; k++)
        for (int l = 0; l < sc.lmax; l++)
            assert(sc.down[l][k] == -1 || (sc.down[l][k] >= 0 && sc.down[l][k] < sc.count));

    int riv_count = rc.count, slo_count = sc.count;
    rri_riv_cellset_free(&rc);
    rri_slo_cellset_free(&sc);
    printf("test_idx_setting: OK (riv_count=%d slo_count=%d)\n", riv_count, slo_count);
    return 0;
}
