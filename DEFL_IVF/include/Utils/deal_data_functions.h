#pragma once

#include<vector>
#include<fstream>
#include<cmath>
#include<algorithm>
#include<fstream>
#include<queue>
#include<set>
#include<unordered_set>
#include<iostream>
#include"def.h"

void read_csv(
    string file_name,
    vector<vector<string>> &res
);

void read_csv(
    string file_name,
    vector<vector<float>> &res
);

vector<string> splitby(
    string s,
    char ch
);

void write_data(
    string file_name,
    vector<vector<float>> &data
);

void write_data(
    string file_name,
    vector<vector<string>> &data
);
void csv_to_data(string file_name);

bool read_data_from_csv(
    string file_name,
    vector<vector<float>> &res
);

bool read_data(
    string file_name,
    vector<vector<float>> &res
);


void write_gt(string file_name,vector<vector<int>> &data);

bool read_gt(string file_name,vector<vector<int>> &res);