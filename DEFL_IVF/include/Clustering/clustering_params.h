#pragma once
#include<cstring>

struct cluster_params 
{
    int* int_param=nullptr;    
    virtual cluster_params* clone()=0;
};

struct kmeans_params:public cluster_params
{
    kmeans_params();
    kmeans_params(int x);
    kmeans_params(int nlist,int nredo,int niter,int max_points);
    kmeans_params(kmeans_params& other);
    kmeans_params* clone() override;
    ~kmeans_params();
};

struct Clustering
{
    virtual float* clustering(float *data, int n,int dim,bool do_sample)=0;
};