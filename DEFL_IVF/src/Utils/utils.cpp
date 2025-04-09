#include"Utils/utils.h"

double recall(vector<int> &base,vector<int> &cmp)
{
    unordered_set<int> S;
    int fz=0;
    for(int i:base) S.insert(i);
    for(int i:cmp)
        if(S.count(i))
            fz++;
    return 1.0*fz/base.size();
}

double recall(vector<PDI> &base,vector<PDI> &cmp)
{
    unordered_set<int> S;
    int fz=0;
    for(auto &[d,i]:base) S.insert(i);
    for(auto &[d,i]:cmp)
        if(S.count(i))
            fz++;
    return 1.0*fz/base.size();
}

double recall(vector<int> &base,vector<PDI> &cmp)
{
    vector<int> cmp_;
    // cmp_.resize(cmp.size());
    for(auto &[u,v]:cmp) cmp_.push_back(v);
    return recall(base,cmp_);
}

double recall(vector<PDI> &base,vector<int> &cmp)
{
    vector<int> base_;
    // base_.resize(base.size());
    for(auto &[u,v]:base) base_.push_back(v);
    return recall(base_,cmp);
}
double recall(vector<int> &base, vector<PDI> &cmp, int k) 
{
    vector<int> base_k(base.begin(), base.begin() + k);
    return recall(base_k, cmp);
}
void rand(int* arr, int count, int max,int seed)
{
    vector<int> numbers(max);
    iota(numbers.begin(), numbers.end(), 0); 
    mt19937 gen(seed);
    shuffle(numbers.begin(), numbers.end(), gen);  
    for (int i = 0; i < count; ++i) 
    {
        arr[i] = numbers[i];
    }
}

vector<int> flat_vector(vector<vector<int>> &aim)
{
    vector<int> res;
    for(auto &tmp:aim)
    {
        res.insert(res.end(),tmp.begin(),tmp.end());
    }
    return res;
}