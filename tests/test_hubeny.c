/* hubeny_sub geodesic distance: 1 degree of latitude ~ 111.19 km on WGS84. */
#include "rri/rri.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void)
{
    double d = rri_hubeny_sub(110.0, 0.0, 110.0, 1.0);
    printf("1 deg latitude at equator: %.3f m\n", d);
    assert(fabs(d - 110574.0) < 1000.0); /* WGS84 meridian arc, ~110.57 km at the equator */

    double d0 = rri_hubeny_sub(110.0, -8.0, 110.0, -8.0);
    assert(fabs(d0) < 1e-6);

    printf("test_hubeny: OK\n");
    return 0;
}
