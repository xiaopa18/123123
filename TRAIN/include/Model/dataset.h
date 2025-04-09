#pragma once
#include<vector>
#include<cstring>
#include<string>
#include<fstream>
#include<iostream>
#include<random>
#include<algorithm>
using namespace std;

class KnnDataset
{
private:
    vector<vector<int>> dataset;
    vector<int> mp;
public:
    KnnDataset(const string &csv_file);
    pair<int,vector<int>> get(int id);
    void shuffle_mp();
    size_t size();
};