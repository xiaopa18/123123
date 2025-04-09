#pragma once
#include <torch/torch.h>

struct DES_MLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr};

    DES_MLP(int64_t input_dim, int64_t output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, input_dim * 2));
        fc2 = register_module("fc2", torch::nn::Linear(input_dim * 2, input_dim * 2));
        fc3 = register_module("fc3", torch::nn::Linear(input_dim * 2, input_dim));
        fc4 = register_module("fc4", torch::nn::Linear(input_dim, input_dim / 2));
        fc5 = register_module("fc5", torch::nn::Linear(input_dim / 2, output_dim));
    }
    torch::Tensor forward(torch::Tensor x) {
        auto tmp = x.clone();
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = torch::relu(fc3(x) + tmp);
        x = torch::relu(fc4(x));
        x = fc5(x);
        return x;
    }
};
struct ASC_MLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr};

    ASC_MLP(int64_t input_dim, int64_t output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, input_dim * 2));
        fc2 = register_module("fc2", torch::nn::Linear(input_dim * 2, input_dim * 2));
        fc3 = register_module("fc3", torch::nn::Linear(input_dim * 2, input_dim * 4));
        fc4 = register_module("fc4", torch::nn::Linear(input_dim * 4, input_dim * 2));
        fc5 = register_module("fc5", torch::nn::Linear(input_dim * 2, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = torch::relu(fc3(x));
        x = torch::relu(fc4(x));
        x = fc5(x);
        return x;
    }
};
