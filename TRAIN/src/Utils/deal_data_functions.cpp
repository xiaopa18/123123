#include"Utils/deal_data_functions.h"


void read_csv(string file_name,vector<vector<string>> &res)
{
    ifstream inf(file_name,ios::in);
    if(!inf.is_open())
    {
        cout<<file_name<<"can't open\n";
        exit(-1);
    }
    string line;
    while(getline(inf,line))
    {
       string tmp="";
       vector<string> now;
       for(char ch:line)
       {
           if(ch==',')
           {
               if (tmp != "")now.push_back(tmp);
               tmp="";
           }else if(ch != '\n' && ch != '\r') tmp.push_back(ch);
       }
       if(tmp!="") now.push_back(tmp);
       if(now.size()) res.push_back(now);
    }
}

void read_csv(string file_name,vector<vector<float>> &res)
{
    ifstream inf(file_name, ios::in);
    if(!inf.is_open())
    {
        exit(-1);
    }
    string tmp;
    int line = 0;
    while (getline(inf, tmp))
    {
        line++;
        vector<float> data;
        string tp = "";
        for (size_t i = 0; i < tmp.size(); i++)
        {
            if (tmp[i] == ',')
            {
                if (tp != "") data.push_back(stod(tp));
                tp = "";
            }
            else if(tmp[i]!='\n' and tmp[i]!='\r')tp += tmp[i];
        }
        if (tp != "") data.push_back(stod(tp));
        if (res.size() && data.size() != res.back().size())
        {
            exit(-1);
        }
        if(data.size()) res.push_back(data);
    }
    inf.close();
}

vector<string> splitby(string s,char ch)
{
    string tmp="";
    vector<string> res;
    for(char &c:s)
    {
        if(c==ch)
        {
            if(tmp!="") res.push_back(tmp);
            tmp="";
        }else tmp.push_back(c);
    }
    if(tmp!="") res.push_back(tmp);
    return res;
}

void write_data(string file_name,vector<vector<float>> &data)
{
    ofstream ouf(file_name,ios::out|ios::binary);
    int n=data.size(),dim=data[0].size();
    ouf.write((char*)&n,sizeof n);
    ouf.write((char*)&dim,sizeof dim);
    for(int i=0;i<n;i++)
        for(int j=0;j<dim;j++)
            ouf.write((char*)&data[i][j],sizeof data[i][j]);
    ouf.close();
}

void write_data(string file_name,vector<vector<string>> &data)
{
    vector<vector<float>> dt;
    for(int i=0;i<data.size();i++)
    {
        vector<float> tmp;
        for(int j=0;j<data[0].size();j++)
        {
            tmp.push_back(stof(data[i][j]));
        }
        dt.push_back(tmp);
    }
    write_data(file_name,dt);
}

void csv_to_data(string file_name)
{
    vector<vector<string>> data;
    read_csv(file_name,data);
    while(file_name.back()!='.') file_name.pop_back();
    file_name+="data";
    write_data(file_name,data);
}

bool read_data_from_csv(string file_name,vector<vector<float>> &res)
{
    ifstream inf(file_name,ios::in);
    if(!inf.is_open())
    {
        cout<<file_name<<" open failed!"<<endl;
        return false;
    }
    res.clear();
    string tmp;
    bool flag=false;
    while(getline(inf,tmp))
    {
        string fl="";
        vector<float> vec;
        for(char ch:tmp)
        {
            if(ch==',')
            {
                if(fl!="")
                {
                    vec.push_back(stof(fl));
                }
                fl="";
            }else fl.push_back(ch);
        }
        if(fl!="") vec.push_back(stof(fl));
        if(flag and vec.size()!=res.back().size())
        {
            cout<<file_name<<" has data inconsistency!"<<endl;
            return false;
        }else flag=true;
        res.push_back(vec);
    }
    return true;
}

bool read_data(string file_name,vector<vector<float>> &res)
{
    ifstream inf(file_name,ios::in|ios::binary);
    cout<<file_name<<endl;
    if(!inf.is_open())
    {
        cout<<file_name<<" open failed!\n";
        return false;
    }
    int n,dim;
    inf.read((char*)&n,sizeof n);
    inf.read((char*)&dim,sizeof dim);
    for(int i=0;i<n;i++)
    {
        vector<float> tmp(dim,0);
        inf.read((char*)tmp.data(), sizeof(float) * dim);
        res.push_back(tmp);
    }
    return true;
}

void write_gt(string file_name,vector<vector<int>> &data)
{
    ofstream ouf(file_name,ios::out|ios::binary);
    if(!ouf.is_open())
    {
        cout<<file_name<<" open failed!\n";
        exit(-1);
    }
    int n=data.size(),dim;
    ouf.write((char*)&n,sizeof n);
    for(int i=0;i<n;i++)
    {
        dim=data[i].size();
        ouf.write((char*)&dim,sizeof dim);
        for(int j=0;j<dim;j++)
            ouf.write((char*)&data[i][j],sizeof data[i][j]);
    }
    ouf.close();
}


bool read_gt(string file_name,vector<vector<int>> &res)
{
    ifstream inf(file_name,ios::in|ios::binary);
    if(!inf.is_open())
    {
        cout<<file_name<<" open failed!\n";
        return false;
    }
    int n,dim;
    inf.read((char*)&n,sizeof n);
    
    for(int i=0;i<n;i++)
    {
        inf.read((char*)&dim,sizeof dim);
        vector<int> tmp(dim,0);
        inf.read((char*)tmp.data(), sizeof(int) * dim);
        res.push_back(tmp);
    }
    return true;
}