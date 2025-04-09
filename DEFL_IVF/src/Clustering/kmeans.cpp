#include"Clustering/kmeans.h"

kmeans::kmeans(kmeans_params cp):cp(cp){}

float* kmeans::clustering(float *data, int n,int dim,bool do_sample=true)
{
    float *centroids = new float[dim*(cp.int_param[0])];
    float *now_data = data;
    bool sample=false;
    size_t now_n=n;
    size_t nsplit=0;
    size_t d=dim;
    int *belong = new int[now_n];
    if(do_sample and n>this->cp.int_param[3]*this->cp.int_param[0])
    {
        sample = true;
        now_n=this->cp.int_param[3]*this->cp.int_param[0];
        cout<<"now clu n => :"<<now_n<<endl;
        float* x=new float[now_n*dim];
        int *sub_idx=new int[now_n]();
        rand(sub_idx, now_n, n,1234);
        for (size_t i = 0; i < now_n; i++)
        {
            memcpy(x + i * d, data + sub_idx[i] * d, d*sizeof(float));
        }
        now_data=x;
        delete[] sub_idx;
    }
    float best_obj=numeric_limits<float>::max();
    float* best_centroids=new float[dim*(cp.int_param[0])];
    for(int redo = 0; redo < this->cp.int_param[1]; redo++)
    {
        vector<int> perm1(now_n);
        rand(perm1.data(), now_n, now_n,1234);
        for (size_t i = 0; i < this->cp.int_param[0]; i++) {
            memcpy(centroids+i*d, now_data + perm1[i] * d, d*sizeof(float));
        }
        float obj = 0;
        for (size_t i = 0; i < cp.int_param[2]; i++)
        {
            vector<float> hassign(cp.int_param[0]);
            obj=search_clu(now_data,now_n,d,centroids,belong);
            compute_centroids(now_data,now_n,d,centroids,hassign.data(),belong);               
        }
        if(best_obj>obj)
        {
            best_obj=obj;
            memcpy(best_centroids,centroids,cp.int_param[0]*d*sizeof(float));
        }
    }

    if(sample) delete[] now_data;
    delete[] belong;
    delete[] centroids;
    return best_centroids;
}

float kmeans::search_clu(float *data,int n,int d,float* centroids,int *belong)
{
    float obj=0;
    for(size_t i=0;i<n;i++)
    {
        float min_dis=numeric_limits<float>::max();
        float res1=-1;
        for(size_t j=0;j<cp.int_param[0];j++)
        {
            float temp=L2_no_sqrt(data+i*d,centroids+j*d,d);
            if(min_dis>temp)
            {
                min_dis=temp;
                res1=j;
            }
        }
        if(res1==-1)
        {
           return -1;
        } 
        belong[i]=res1;
        obj+=min_dis;
    }
    return obj;
}

void kmeans::compute_centroids(float *data, int n,int d, float *centroids,float *hassign,int *belong)
{

    memset(centroids, 0, d*sizeof(float) * cp.int_param[0]);
    for(size_t i=0;i<n;i++) 
    {
        size_t ci = belong[i];
        float* c = centroids + ci * d;  
        const float* xi = data + i * d;  
        hassign[ci]+=1.0;
        for(size_t j=0;j<d;j++)
        {
            c[j]+=xi[j];
        }
    }
    for(int ci = 0; ci < cp.int_param[0]; ci++) 
    {
        if (hassign[ci] == 0)
        {
            continue; 
        }
        float norm = 1 / hassign[ci];  
        float* c = centroids + ci * d; 
        for (int j = 0; j < d; j++) 
        {
            c[j] *= norm;  
        }
    }
}