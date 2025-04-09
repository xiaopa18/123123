#pragma once
#include "Model/model.h"
#include "Model/loss_function.h"
#include "Model/dataset.h"
#include "Utils/utils.h"
#include <torch/torch.h>
#include<thread>
#include<iostream>
#include<chrono>
using namespace std;
using namespace std::chrono;

void single_train(
    int epoch, 
    size_t cmp_cnt,
    shared_ptr<DES_MLP> &model,
    torch::Tensor &data, 
    KnnDataset &trainset,
    torch::Device &device
);

void parallel_train_function_contrastive_loss(
    shared_ptr<DES_MLP> model,
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device,
    double *loss_val,
    float *ratio
);

void parallel_train_function_self_loss(
    shared_ptr<DES_MLP> model,
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device,
    double *loss_val
);

void parallel_contrastive_train(
    int epoch,
    size_t cmp_cnt,
    shared_ptr<DES_MLP> model,
    torch::Tensor &data, 
    KnnDataset &trainset,
    vector<int> &device_id,
    int in_dim,
    int out_dim,
    string save_path,
    float ratio
);

void parallel_self_train(
    int epoch,
    size_t cmp_cnt,
    shared_ptr<DES_MLP> model,
    torch::Tensor &data, 
    KnnDataset &trainset,
    vector<int> &device_id,
    int in_dim,
    int out_dim,
    string save_path
);