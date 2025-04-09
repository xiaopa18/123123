#include"Index/ivf.h"
ivf::ivf(int n,int origin_dim,float *dataset,int lite_dim,float *trans_dataset,cluster_params *cp,
         bool can_sample):n(n),origin_dim(origin_dim),lite_dim(lite_dim),sum_dim(origin_dim+lite_dim),
         clu_num(cp->int_param[0]),clu_rank(clu_num)
{
    nprobe = 1;
    origin_data=vector<vector<float>>(clu_num,vector<float>());
    lite_data=vector<vector<float>>(clu_num,vector<float>());
    idmp=vector<vector<int>>(clu_num,vector<int>());
    nums=vector<int>(clu_num,0);
    kmeans km(*(kmeans_params*)cp);
    float *centers=km.clustering(trans_dataset,n,lite_dim,can_sample); 
    centroids.resize(clu_num*lite_dim);
    memcpy(centroids.data(), centers, clu_num*lite_dim*sizeof(float));
    delete[] centers;
    int max_num=0;
    size_t pp=n/10;
    for(size_t i=0;i<n;i++)
    {
        clu_ranking(trans_dataset+i*lite_dim); 
        for(size_t j=0;j<lite_dim;j++)
        {
            lite_data[clu_rank[0].se].push_back(trans_dataset[i*lite_dim+j]);
        } 
        for(size_t j=0;j<origin_dim;j++)
        {
            origin_data[clu_rank[0].se].push_back(dataset[i*origin_dim+j]); 
        }
        idmp[clu_rank[0].se].push_back(i);
        inv_idmp[i]={clu_rank[0].se,nums[clu_rank[0].se]}; 
        nums[clu_rank[0].se]++;
        max_num = max(max_num,nums[clu_rank[0].se]); 
    }
    distance_register.resize(max_num);
}
void ivf::clu_ranking(float *query)
{
    float *center = centroids.data();
    for(int i=0;i<clu_num;i++,center+=lite_dim)
        clu_rank[i]={L2_no_sqrt(center,query,lite_dim),i};
    sort(clu_rank.begin(),clu_rank.end());
}

vector<int>* ivf::range_query(float *query,float *query_lite, float r,float ratio)
{
    int *id_mp;
    vector<int> *res=new vector<int>();
    float r2=r*r,ratio2=ratio*ratio,approx_r2,*o_data,*l_data,*reg=distance_register.data();
    clu_ranking(query_lite);
    for(size_t clu_id=0;clu_id<nprobe;clu_id++)
    {
        size_t real_clu_id = clu_rank[clu_id].se;
        o_data=origin_data[real_clu_id].data();
        l_data=lite_data[real_clu_id].data();
        id_mp=idmp[real_clu_id].data();
        for(size_t i=0;i<nums[real_clu_id];i++,l_data+=lite_dim)
        {
            reg[i]=L2_no_sqrt(l_data,query_lite,lite_dim);
        }
        for(size_t i=0;i<nums[real_clu_id];i++)
        {
            if(r2*ratio2<reg[i])
            {
                continue;
            }
            approx_r2 = L2_no_sqrt(o_data+i*origin_dim,query,origin_dim);
            if(approx_r2 <= r2)
                res->push_back(id_mp[i]);
        }
    }
    return res;
}

priority_queue<PDI>* ivf::knn(float *query,float *query_lite, int k, float ratio)
{
    int *id_mp;
    size_t search_cnt=0,clu_id=0;
    float r2=numeric_limits<float>::max();
    float ratio2=ratio*ratio,approx_r2,*l_data,*o_data,*reg=distance_register.data();
    priority_queue<PDI> *res = new priority_queue<PDI>();
    unordered_set<int> c1_cand;
    for(size_t i=0;i<k;i++) res->push({r2,-1});
    clu_ranking(query_lite);

    for(clu_id=0;clu_id<nprobe;clu_id++)
    {
        size_t real_clu_id = clu_rank[clu_id].se;
        l_data = lite_data[real_clu_id].data();
        id_mp = idmp[real_clu_id].data();
        for(size_t i=0;i<nums[real_clu_id];i++,l_data+=lite_dim)
        {
            reg[i] = L2_no_sqrt(l_data,query_lite,lite_dim);
            res->push({reg[i],id_mp[i]});
            res->pop();
        }
        search_cnt += nums[real_clu_id];
        if(search_cnt>=k) break;
    }
    if(search_cnt<k)
    {
        delete res;
        res = new priority_queue<PDI>();
        for(size_t i=0;i<k;i++) res->push({r2,-1});
        for(size_t i=0;i<nprobe;i++)
        {
            size_t real_clu_id = clu_rank[i].se;
            o_data = origin_data[real_clu_id].data();
            id_mp = idmp[real_clu_id].data();
            for(size_t i=0;i<nums[real_clu_id];i++,o_data+=origin_dim)
            {
                res->push({L2_no_sqrt(o_data,query,origin_dim),id_mp[i]});
                res->pop();
            }
        }
        return res;
    }
    for(size_t i=0;i<k;i++)
    {
        c1_cand.insert(res->top().se);
        res->pop();
    }

    for(auto i:c1_cand)
    {
        auto [clu_id,inid]=inv_idmp[i];
        res->push({L2_no_sqrt(origin_data[clu_id].data()+inid*origin_dim,query,origin_dim),i});
    }
    r2=res->top().fi;

    for(size_t i=0;i<=clu_id;i++)
    {
        size_t real_clu_id = clu_rank[i].se;
        o_data = origin_data[real_clu_id].data();
        id_mp = idmp[real_clu_id].data();
        for(size_t i=0;i<nums[real_clu_id];i++)
        {
            if(c1_cand.count(id_mp[i])) continue;
            if(r2 * ratio2 < reg[i])
            {
                continue;
            } 
            approx_r2 = L2_no_sqrt(o_data+i*origin_dim,query,origin_dim);
            if(approx_r2 > r2) continue;
            res->push({approx_r2,id_mp[i]});
            res->pop();
            r2 = res->top().fi;
        }
    }

    for(clu_id++;clu_id<nprobe;clu_id++)
    {
        size_t real_clu_id = clu_rank[clu_id].se;
        l_data = lite_data[real_clu_id].data();
        o_data = origin_data[real_clu_id].data();
        id_mp = idmp[real_clu_id].data();
        for(size_t i=0;i<nums[real_clu_id];i++,l_data+=lite_dim)
        {
            reg[i] = L2_no_sqrt(l_data,query_lite,lite_dim);
        }
        for(size_t i=0;i<nums[real_clu_id];i++)
        {
            if(r2 * ratio2 < reg[i])
            {
                continue;
            } 
            
            approx_r2 = L2_no_sqrt(o_data+i*origin_dim,query,origin_dim);
            if(approx_r2 > r2) continue;
            res->push({approx_r2,id_mp[i]});
            res->pop();
            r2 = res->top().fi;
        }
    }
    return res;
}

vector<PDI>* ivf::exact_range_query(float *query, float *query_lite, float r)
{
    int *id_mp;
    vector<PDI> *res=new vector<PDI>();
    float r2=r*r,approx_r2,*o_data;
    clu_ranking(query_lite);
    for(size_t clu_id=0;clu_id<nprobe;clu_id++)
    {
        size_t real_clu_id = clu_rank[clu_id].se;
        o_data=origin_data[real_clu_id].data();
        id_mp=idmp[real_clu_id].data();
        for(size_t i=0;i<nums[real_clu_id];i++)
        {
            approx_r2 = L2_no_sqrt(o_data+i*origin_dim,query,origin_dim);
            if(approx_r2 <= r2)
                res->push_back({approx_r2,id_mp[i]});
        }
    }
    return res;
}

priority_queue<PDI>* ivf::exact_knn(float *query, float *query_lite,int k)
{
    int *id_mp,search_cnt=0;
    float r2=numeric_limits<float>::max(),approx_r2,*o_data;
    priority_queue<PDI> *res=new priority_queue<PDI>();
    for(size_t i=0;i<k;i++) res->push({r2,-1});
    clu_ranking(query_lite);
    for(size_t i=0;i<nprobe;i++)
    {
        size_t real_clu_id = clu_rank[i].se;
        o_data = origin_data[real_clu_id].data();
        id_mp = idmp[real_clu_id].data();
        for(size_t i=0;i<nums[real_clu_id];i++)
        {
            approx_r2 = L2_no_sqrt(o_data+i*origin_dim,query,origin_dim);
            if(approx_r2 > r2) continue;
            res->push({approx_r2,id_mp[i]});
            res->pop();
            r2 = res->top().fi;
        }
    }
    return res;
}

void ivf::set_nprobe(int nprobe)
{
    this->nprobe = nprobe;
}