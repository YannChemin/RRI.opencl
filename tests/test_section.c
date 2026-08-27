/* hr2vr/vr2hr are inverses for the rectangular-channel fallback. */
#include "rri/rri.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void)
{
    double area = 100.0;
    double area_ratio = 0.3;
    for (double hr = 0.0; hr <= 5.0; hr += 0.37) {
        double vr = rri_hr2vr(hr, area, area_ratio);
        double hr2 = rri_vr2hr(vr, area, area_ratio);
        assert(fabs(hr - hr2) < 1e-9);
    }
    assert(fabs(rri_hr2vr(0.0, area, area_ratio)) < 1e-12);
    printf("test_section: OK\n");
    return 0;
}
