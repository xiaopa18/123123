#include "Model/dataset.h"

KnnDataset::KnnDataset(const string &csv_file)
{
    ifstream inf(csv_file,ios::in);
    if(!inf.is_open())
    {
        cout<<"open "<<csv_file<<" is failed!"<<endl;
        exit(-1);
    }
    int cnt=0;
    string line;
    bool fir=false;
    while(getline(inf,line))
    {
        vector<int> res;
        string tmp="";
        bool fir=0;
        for(int i=0;i<line.size();i++)
        {
            if(line[i]==',')
            {
                if(!fir)
                {
                    fir=true;
                    tmp="";
                    continue;
                }
                if(tmp!="") res.push_back(stoi(tmp));
                tmp="";
            }
            else if(line[i]>='0' and line[i]<='9')
            {
                tmp.push_back(line[i]);
            }
        }
        if(tmp!="") 
        {
            res.push_back(stoi(tmp));
        }
        dataset.push_back(res);
        mp.push_back(cnt++);
    }
}

pair<int,vector<int>> KnnDataset::get(int id)
{
    return make_pair(mp[id],dataset[mp[id]]);
}

size_t KnnDataset::size()
{
    return dataset.size();
}

void KnnDataset::shuffle_mp()
{
    random_device rd; 
    mt19937 g(rd());  
    shuffle(mp.begin(), mp.end(), g);
}