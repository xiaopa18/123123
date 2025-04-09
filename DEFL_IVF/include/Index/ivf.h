#pragma once
#include"Utils/space.h"
#include"Utils/utils.h"
#include"Clustering/clustering_params.h"
#include"Clustering/kmeans.h"
#include<numeric>
#include<vector>
#include<queue>
#include<cstdlib>
#include<iostream>
#include<set>
#include<algorithm>
#include<unordered_set>
#include<unordered_map>
#include"def.h"
using namespace std;
struct ivf
{
    int n;
    vector<int> nums;
    int sum_dim;
    int lite_dim;
    vector<vector<float>> lite_data;
    int origin_dim;
    vector<vector<float>> origin_data;
    int nprobe;
    int clu_num;
    vector<float> distance_register;
    vector<vector<int>> idmp;
    vector<float> centroids;
    vector<PDI> clu_rank;
    unordered_map<int,PII> inv_idmp;
    ivf(int n,int origin_dim,float *dataset,int lite_dim,float *trans_dataset,cluster_params *cp, bool can_sample=true);
    void clu_ranking(float *query);
    vector<int>* range_query(float *query, float *query_lite, float r, float ratio=1);
    priority_queue<PDI>* knn(float *query, float *query_lite, int k, float ratio=1);
    vector<PDI>* exact_range_query(float *query, float *query_lite, float r);
    priority_queue<PDI>* exact_knn(float *query, float *query_lite,int k);
    void set_nprobe(int nprobe);
};