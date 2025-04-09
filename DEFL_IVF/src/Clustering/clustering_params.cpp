#include"Clustering/clustering_params.h"


kmeans_params::kmeans_params() 
{
    size_t num_elements = 4;
    int_param = new int[num_elements];
    int_param[0]=10;
    int_param[1]=1;
    int_param[2]=10; 
    int_param[3]=256;
}
kmeans_params::kmeans_params(int x) 
{
    size_t num_elements = 4;
    int_param = new int[num_elements];
    int_param[0]=x;
    int_param[1]=1;
    int_param[2]=10;
    int_param[3]=256;
}
kmeans_params::kmeans_params(int nlist,int nredo,int niter,int max_points) 
{
    size_t num_elements = 4;
    int_param = new int[num_elements];
    int_param[0]=nlist;
    int_param[1]=nredo;
    int_param[2]=niter; 
    int_param[3]=max_points;
}
kmeans_params::kmeans_params(kmeans_params& other) 
{
    size_t num_elements = 4; 
    int_param = new int[num_elements];
    memcpy(int_param, other.int_param, sizeof(int) * num_elements);
}
kmeans_params::~kmeans_params() 
{
    delete[] int_param;
}

kmeans_params* kmeans_params::clone()
{
    return new kmeans_params(*this);
}
