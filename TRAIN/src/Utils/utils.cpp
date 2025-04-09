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
    for(auto &[u,v]:cmp) cmp_.push_back(v);
    return recall(base,cmp_);
}

double recall(vector<PDI> &base,vector<int> &cmp)
{
    vector<int> base_;
    for(auto &[u,v]:base) base_.push_back(v);
    return recall(base_,cmp);
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

void thread_calc_purnning(
    int st,
    int ed,
    int thread_id,
    vector<PDD> *ex,
    vector<PDD> *in,
    int n,int nq,float r,float *dataset, 
    float *queryset, float *trans_dataset,
    float* trans_queryset,int dim,int trans_dim
)
{
    double ex_ans=0,ex_cnt=0;
    double in_ans=0,in_cnt=0;
    float tmp,tmp2,r2=r*r;
    for(int i=st;i<ed;i++)
    {
        for(int j=0;j<n;j++)
        {
            tmp=L2_no_sqrt(dataset+j*dim,queryset+i*dim,dim);
            tmp2=L2_no_sqrt(trans_dataset+j*trans_dim, 
                            trans_queryset+i*trans_dim,trans_dim);
            if(tmp<=r2)
            {
                in_cnt++;
                if(tmp2<=r2) in_ans++;
            }
            else
            {
                ex_cnt++;
                if(tmp2>r2) ex_ans++;
            }
        }
    }
    (*in)[thread_id]=make_pair(in_ans,in_cnt);
    (*ex)[thread_id]=make_pair(ex_ans,ex_cnt);
}

void calc_purnning(int thread_num,int n,int nq,float r,float *dataset, 
                float *queryset, float *trans_dataset,
                float* trans_queryset,int dim,int trans_dim)
{
    int skip=(nq+thread_num-1)/thread_num;
    vector<PDD> in(thread_num),ex(thread_num);
    vector<thread*> thread_vector;
    double ex_ans=0,ex_cnt=0;
    double in_ans=0,in_cnt=0;
    for(int i=0,id=0;i<nq;i+=skip,id++)
    {
        thread_vector.push_back(new thread(thread_calc_purnning,i,
                                            min(i+skip,nq),id
                                            ,&ex,&in,
                                            n,nq,r,dataset, 
                                            queryset,trans_dataset,
                                            trans_queryset,dim,trans_dim));
    }
    for(int i=0;i<thread_vector.size();i++)
    {
        thread_vector[i]->join();
        in_ans+=in[i].fi;
        in_cnt+=in[i].se;
        ex_ans+=ex[i].fi;
        ex_cnt+=ex[i].se;
    }
    cout<<"in:"<<in_ans/in_cnt<<endl;
    cout<<"ex:"<<ex_ans/ex_cnt<<endl;
}



void thread_calc_ratio(
    int st,
    int ed,
    int thread_id,
    vector<PDD> *ex,
    vector<PDD> *in,
    int n,int nq,float r,float *dataset, 
    float *queryset, float *trans_dataset,
    float* trans_queryset,int dim,int trans_dim
)
{
    double ex_ans=0,ex_cnt=0;
    double in_ans=0,in_cnt=0;
    float tmp,tmp2,r2=r*r;
    for(int i=st;i<ed;i++)
    {
        for(int j=0;j<n;j++)
        {
            tmp=L2_no_sqrt(dataset+j*dim,queryset+i*dim,dim);
            tmp2=L2_no_sqrt(trans_dataset+j*trans_dim, 
                            trans_queryset+i*trans_dim,trans_dim);
            if(tmp<=r2)
            {
                in_cnt++;
                if(tmp2<=r2) in_ans++;
            }
            else
            {
                ex_cnt++;
                if(tmp2>r2) ex_ans++;
            }
        }
    }
    (*in)[thread_id]=make_pair(in_ans,in_cnt);
    (*ex)[thread_id]=make_pair(ex_ans,ex_cnt);
}

void calc_ratio(int thread_num,int n,int nq,float r,float *dataset, 
                float *queryset, float *trans_dataset,
                float* trans_queryset,int dim,int trans_dim)
{
    int skip=(nq+thread_num-1)/thread_num;
    vector<PDD> in(thread_num),ex(thread_num);
    vector<thread*> thread_vector;
    double ex_ans=0,ex_cnt=0;
    double in_ans=0,in_cnt=0;
    for(int i=0,id=0;i<nq;i+=skip,id++)
    {
        thread_vector.push_back(new thread(thread_calc_purnning,i,
                                            min(i+skip,nq),id
                                            ,&ex,&in,
                                            n,nq,r,dataset, 
                                            queryset,trans_dataset,
                                            trans_queryset,dim,trans_dim));
    }
    for(int i=0;i<thread_vector.size();i++)
    {
        thread_vector[i]->join();
        in_ans+=in[i].fi;
        in_cnt+=in[i].se;
        ex_ans+=ex[i].fi;
        ex_cnt+=ex[i].se;
    }
    cout<<"in:"<<in_ans/in_cnt<<endl;
    cout<<"ex:"<<ex_ans/ex_cnt<<endl;
}