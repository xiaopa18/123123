#include"Index/enum.h"
double enumeration::memory_used()
{
    return sizeof(int)*2+sizeof(float)*dataset.size();
}

enumeration::enumeration(vector<vector<float>> dataset)
{
    num=dataset.size();
    dim=dataset[0].size();
    this->dataset.resize(num*dim);
    for(int i=0,cnt=0;i<dataset.size();i++)
    {
        memcpy(this->dataset.data()+i*dim, dataset[i].data(), sizeof(float)*dim);
    }
}
enumeration::enumeration(torch::Tensor dataset)
{
    num=dataset.sizes()[0], dim=dataset.sizes()[1];
    this->dataset.resize(num*dim);
    memcpy(this->dataset.data(), dataset.data_ptr<float>(), sizeof(float)*num*dim);
}
bool enumeration::insert(vector<float> p)
{
    for(int i=0;i<dim;i++)
        dataset.push_back(p[i]);
    num++;
    return true;
}
bool enumeration::del(int id)
{
    dataset.erase(dataset.begin()+id*dim,dataset.begin()+(id+1)*dim);
    num--;
    return true;
}
vector<PDI>* enumeration::range_query(float* q,float r)
{
    double r2=r*r;
    vector<PDI> *res=new vector<PDI>();
    for(int i=0;i<num;i++)
    {
        double tmp=L2_no_sqrt(q,dataset.data()+dim*i,dim);
        if(tmp>r2) continue;
        res->push_back({tmp,i});
    }
    return res;
}
vector<PDI>* enumeration::range_query(float* q,double r,vector<int> *cand)
{
    double r2=r*r;
    vector<PDI> *res=new vector<PDI>();
    for(int i:*cand)
    {
        double tmp=L2_no_sqrt(q,dataset.data()+dim*i,dim);
        if(tmp>r2) continue;
        res->push_back({tmp,i});
    }
    return res;
}

vector<int>* enumeration::range_query_(float* q,double r)
{
    double r2=r*r;
    vector<int> *res=new vector<int>();
    for(int i=0;i<num;i++)
    {
        double tmp=L2_no_sqrt(q,dataset.data()+dim*i,dim);
        if(tmp>r2) continue;
        res->push_back(i);
    }
    return res;
}
priority_queue<PDI>* enumeration::knn(float* q,int k)
{
    priority_queue<PDI>* res=new priority_queue<PDI>();
    double r2=1e20;
    for(int i=0;i<num;i++)
    {
        double tmp=L2_no_sqrt(q,dataset.data()+dim*i,dim);
        if(tmp>r2) continue;
        res->push({tmp,i});
        if(res->size()>k)
        {
            res->pop();
            r2=res->top().fi;
        }
    }
    return res;
}
double enumeration::get_max_distance(float *q,vector<int> *cand)
{
    double res=0;
    for(int i:*cand)
    {
        double tmp=L2_no_sqrt(q,dataset.data()+dim*i,dim);
        res=max(res,tmp);
    }
    return res;        
}

vector<PDI>* enumeration::get_distance(float* q, vector<int> *cand)
{
    int idx = 0;
    vector<PDI> *res=new vector<PDI>(cand->size(),{0,0});
    for(int i:*cand)
    {
        double tmp=L2_no_sqrt(q,dataset.data()+dim*i,dim);
        (*res)[idx++]={tmp,i};
    }
    return res;
}

double enumeration::get_max_distance(float *q,priority_queue<PDI> *cand)
{
    double res=0;
    while(!cand->empty())
    {
        double tmp=L2_no_sqrt(q,dataset.data()+dim*cand->top().se,dim);
        cand->pop();
        res=max(res,tmp);
    }
    return res;
}