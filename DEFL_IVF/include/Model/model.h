#pragma once
#include <torch/torch.h>
#include <bits/stdc++.h>
struct DES_MLP : torch::nn::Module 
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr};
    DES_MLP(int64_t input_dim, int64_t output_dim);
    torch::Tensor forward(torch::Tensor x);
};
