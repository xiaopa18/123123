#pragma once
#include<random>
#include<numeric>
#include<algorithm>
#include<unordered_set>
#include"def.h"
using namespace std;

void rand(int* arr, int count, int max,int seed);
vector<int> flat_vector(vector<vector<int>> &aim);

double recall(
    vector<int> &base,
    vector<int> &cmp
);
double recall(vector<int> &base, vector<PDI> &cmp, int k) ;

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