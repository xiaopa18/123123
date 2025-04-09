#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include<vector>
#include"Utils/space.h"
#include"def.h"
#include<torch/torch.h>

using namespace std;

struct enumeration
{
    int dim,num;
    vector<float> dataset;
    double memory_used();
    enumeration(vector<vector<float>> dataset);
    enumeration(torch::Tensor dataset);
    bool insert(vector<float> p);
    bool del(int id);
    vector<PDI>* range_query(float* q,float r);
    vector<PDI>* range_query(float* q,double r,vector<int> *cand);
    vector<PDI>* get_distance(float* q, vector<int> *cand);
    vector<int>* range_query_(float* q,double r);
    priority_queue<PDI>* knn(float* q,int k);
    double get_max_distance(float *q,vector<int> *cand);
    double get_max_distance(float *q,priority_queue<PDI> *cand);
};