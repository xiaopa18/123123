#pragma once
#include<random>
#include<numeric>
#include<algorithm>
#include<unordered_set>
#include<iostream>
#include<thread>
#include"Utils/space.h"
#include"def.h"
using namespace std;

void rand(int* arr, int count, int max,int seed);
vector<int> flat_vector(vector<vector<int>> &aim);

double recall(
    vector<int> &base,
    vector<int> &cmp
);

double recall(
    vector<PDI> &base,
    vector<PDI> &cmp
);

double recall(
    vector<int> &base,
    vector<PDI> &cmp
);

double recall(
    vector<PDI> &base,
    vector<int> &cmp
);

void thread_calc_purnning(
    int st,
    int ed,
    int thread_id,
    vector<PDD> *ex,
    vector<PDD> *in,
    int n,int nq,float r,float *dataset, 
    float *queryset, float *trans_dataset,
    float* trans_queryset,int dim,int trans_dim
);

void calc_purnning(int thread_num,int n,int nq,float r,float *dataset, 
                float *queryset, float *trans_dataset,
                float* trans_queryset,int dim,int trans_dim);