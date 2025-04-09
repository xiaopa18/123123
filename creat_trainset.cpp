#include <bits/stdc++.h>
#include "omp.h"
using namespace std;
size_t n, dim, trans_dim, nq;
double max_val=0;
float resres=0;
int k;
size_t k1,k2;
string dataid;
double max_dr=0;
vector<double> rs; 
vector<int> pinlv;
vector<vector<float>> dataset;
vector<vector<float>> queryset;
vector<vector<int>> nntrain;
float L2_no_sqrt(const float *X,const float *Y,int d)
{
    float dis=0;
    for(int k=0;k<d;k++)
    {
        const float* x=X+k;
        const float* y=Y+k;
        dis+=(*x-*y)*(*x-*y);
    }
    return dis;
}
void read_data(const std::string& file_name, std::vector<std::vector<float>>& data) {
    std::ifstream inf(file_name, std::ios::in | std::ios::binary);
    std::cout << file_name << std::endl;
    if (!inf.is_open()) {
        std::cout << file_name << " open failed!\n";
        exit(-1);
    }
    int n, dim;
    inf.read(reinterpret_cast<char*>(&n), sizeof(n));
    inf.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    data.resize(n, std::vector<float>(dim));
    for (int i = 0; i < n; ++i) {
        inf.read(reinterpret_cast<char*>(data[i].data()), sizeof(float) * dim);
    }
}
void trainKnn(size_t k1,size_t k2) 
{
    nntrain.resize(n+5, vector<int>(k2+5));
    rs.resize(n + 100);
    vector<float> local_nntrain(n+5, 0);
#pragma omp parallel for schedule(dynamic, 1000)
    for (size_t i = 0; i < n; i++)
    {
        priority_queue<pair<float, int>> dl;
        for (size_t j = 0; j < n; j++) 
        {
            float dist = L2_no_sqrt(dataset[j].data(), dataset[i].data(), dim);
            
            if (dl.size() < k2) 
            {
                dl.push({dist, j});
            }
            else if (dist < dl.top().first) 
            {
                dl.pop();
                dl.push({dist, j});
            }
        }
        vector<pair<float, int>> temp(k2);
        for (int m = k2 - 1; m >= 0; --m) 
        {
            temp[m] = dl.top();
            dl.pop();
        }
        for (int m = 0; m < k2; ++m) 
        {
            nntrain[i][m] = temp[m].second;
        }
        rs[i] = sqrt(temp[k2 - 1].first) - sqrt(temp[k1 - 1].first);
    }
    for (size_t i = 0; i < n; i++)
    {
        max_dr = max(max_dr, rs[i]);
    }

}
int main(int argc, char **argv)
{  
    dataid=argv[1];
    k1=stoi(argv[2]);
    k2=k1*2;
    omp_set_num_threads(8);
    string data_dir = "/your/path/";
    read_data(data_dir+dataid+"/"+dataid+".data",dataset);
    read_data(data_dir+dataid+"/"+dataid+"_uniform1000.data",queryset);
    n=dataset.size();
    dim=dataset[0].size();
    nq=queryset.size();
    cout<<n<<" "<<dim<<" "<<nq<<endl;
    float max_r1=0;
    cout<<data_dir+dataid+"/"+dataid+"_nntrain.csv"<<endl;
    float max_r2=0;
    trainKnn(k1,k2);
    ofstream ft(dataid+"_train.csv");
    for(size_t i=0;i<n;i++)
    {
        int res=k1;
        res+=(ceil((k2-k1)*(1.0-rs[i]/max_dr)));
        ft<<i;
        for(size_t j=0;j<res;j++)
        {
            ft<<","<<nntrain[i][j];
            if(i==0)
            {
                cout<<","<<nntrain[i][j];
            }
        }ft<<endl;
    }

    return 0;
}
