#pragma once
#include <torch/torch.h>
#include "Model/model.h"

using namespace std;

torch::Tensor log_contrastive_loss(
    shared_ptr<DES_MLP> model, 
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device
);
torch::Tensor abs_contrastive_loss(
    shared_ptr<DES_MLP> model, 
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device
);
torch::Tensor log_abs_contrastive_loss(
    std::shared_ptr<DES_MLP> model,
    torch::Tensor* base,
    torch::Tensor* cmp,
    torch::Tensor* data,
    torch::Device* device,
    float *ratio  
);
torch::Tensor log_self_loss(
    shared_ptr<DES_MLP> model, 
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device
);
torch::Tensor abs_self_loss(
    shared_ptr<DES_MLP> model, 
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device
);