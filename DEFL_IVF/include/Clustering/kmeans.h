#pragma once
#include "Utils/space.h"
#include "Utils/utils.h"
#include "Clustering/clustering_params.h"
#include<iostream>
using namespace std;

struct kmeans: public Clustering
{
    kmeans_params cp;
    kmeans(kmeans_params cp);
    float search_clu(float *data,int n,int d,float *centroids,int *belong);
    float* clustering(float *data, int n,int dim,bool do_sample) override;
    void compute_centroids(float *data, int n,int d,float *centorids,float *hassign,int *belong);
};