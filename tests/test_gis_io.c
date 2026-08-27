/* read_gis_real/read_gis_int round-trip + header-mismatch rejection. */
#include "rri/rri.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void write_grid(const char *path, int comma)
{
    FILE *f = fopen(path, "w");
    fprintf(f, "ncols %d\n", 3);
    fprintf(f, "nrows %d\n", 2);
    fprintf(f, "xllcorner %f\n", 100.0);
    fprintf(f, "yllcorner %f\n", 10.0);
    fprintf(f, "cellsize %f\n", 0.5);
    fprintf(f, "NODATA_value -9999\n");
    if (comma) {
        fprintf(f, "1.5, 2.5, 3.5\n");
        fprintf(f, "-9999, 5.5, 6.5\n");
    } else {
        fprintf(f, "1.5 2.5 3.5\n");
        fprintf(f, "-9999 5.5 6.5\n");
    }
    fclose(f);
}

int main(void)
{
    const char *path_ws = "/tmp/rri_test_grid_ws.asc";
    const char *path_comma = "/tmp/rri_test_grid_comma.asc";
    write_grid(path_ws, 0);
    write_grid(path_comma, 1);

    double out_ws[6], out_comma[6];
    assert(rri_read_gis_real(path_ws, 2, 3, 100.0, 10.0, 0.5, out_ws) == 0);
    assert(rri_read_gis_real(path_comma, 2, 3, 100.0, 10.0, 0.5, out_comma) == 0);
    for (int i = 0; i < 6; i++) assert(fabs(out_ws[i] - out_comma[i]) < 1e-9);
    assert(fabs(out_ws[0] - 1.5) < 1e-9);
    assert(fabs(out_ws[3] - (-9999.0)) < 1e-9);
    assert(fabs(out_ws[5] - 6.5) < 1e-9);

    int ny, nx; double xll, yll, cs;
    assert(rri_read_gis_header(path_ws, &ny, &nx, &xll, &yll, &cs) == 0);
    assert(ny == 2 && nx == 3);
    assert(fabs(xll - 100.0) < 1e-6 && fabs(yll - 10.0) < 1e-6 && fabs(cs - 0.5) < 1e-6);

    /* header mismatch must be rejected */
    assert(rri_read_gis_real(path_ws, 2, 3, 999.0, 10.0, 0.5, out_ws) != 0);
    assert(rri_read_gis_real(path_ws, 5, 3, 100.0, 10.0, 0.5, out_ws) != 0);

    int iout[6];
    assert(rri_read_gis_int(path_ws, 2, 3, 100.0, 10.0, 0.5, iout) == 0);
    assert(iout[3] == -9999);

    printf("test_gis_io: OK\n");
    return 0;
}
