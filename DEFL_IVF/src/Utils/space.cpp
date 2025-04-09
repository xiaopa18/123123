#include"Utils/space.h"
float L2_no_sqrt(float *a, float *b, int dim)
{
    float res=0,tmp=0;
    // #ifdef CALC_DIST_COUNT
    dcmp+=dim;
    // #endif // CALC_DIST_COUNT
    for(int i=0;i<dim;i++)
    {
        tmp=a[i]-b[i];
        res += tmp * tmp;
    }
    return res;
}

float L2_sqrt(float *a, float *b, int dim)
{
    float res=0,tmp=0;
    // #ifdef CALC_DIST_COUNT
    dcmp+=dim;
    // #endif // CALC_DIST_COUNT
    for(int i=0;i<dim;i++)
    {
        tmp=a[i]-b[i];
        res += tmp * tmp;
    }
    return sqrt(res);
}