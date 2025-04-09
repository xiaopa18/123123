#include "Model/loss_function.h"

torch::Tensor log_contrastive_loss(
    shared_ptr<DES_MLP> model, 
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data,
    torch::Device *device
)
{
    model->to((*device));
    auto a = (*data).index({(*base)}).clone().to((*device));  
    auto b = (*data).index({(*cmp)}).clone().to((*device)); 
    auto f_a = model->forward(a);  
    auto f_b = model->forward(b);  

    auto f_a_expanded = f_a.unsqueeze(1); 
    auto f_b_expanded = f_b.unsqueeze(0);  

    auto l2_norms_f = torch::norm(f_a_expanded - f_b_expanded, 2, 2); 
    auto l2_norms_o = torch::norm(a.unsqueeze(1) - b.unsqueeze(0), 2, 2);  

    auto log_l2_norms_f = torch::log(l2_norms_f + 1e-8);  
    auto log_l2_norms_o = torch::log(l2_norms_o + 1e-8);  

    auto loss = torch::mean(torch::pow(log_l2_norms_f - log_l2_norms_o, 2));
    return loss;
}
torch::Tensor abs_contrastive_loss(
    shared_ptr<DES_MLP> model, 
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data,
    torch::Device *device
)
{
    model->to((*device));
    auto a = (*data).index({(*base)}).clone().to((*device)); 
    auto b = (*data).index({(*cmp)}).clone().to((*device));  
    auto f_a = model->forward(a); 
    auto f_b = model->forward(b);  

    auto f_a_expanded = f_a.unsqueeze(1);  
    auto f_b_expanded = f_b.unsqueeze(0); 

    auto l2_norms_f = torch::norm(f_a_expanded - f_b_expanded, 2, 2);
    auto l2_norms_o = torch::norm(a.unsqueeze(1) - b.unsqueeze(0), 2, 2); 

    auto loss = torch::mean(torch::pow(l2_norms_f - l2_norms_o, 2));
    return loss;
}
torch::Tensor log_abs_contrastive_loss(
    std::shared_ptr<DES_MLP> model,
    torch::Tensor* base,
    torch::Tensor* cmp,
    torch::Tensor* data,
    torch::Device* device,
    float *ratio  
) {
    model->to(*device);

    auto a = data->index({*base}).clone().to(*device); 
    auto b = data->index({*cmp}).clone().to(*device);   

    auto f_a = model->forward(a);  
    auto f_b = model->forward(b); 

    auto f_a_expanded = f_a.unsqueeze(1);  
    auto f_b_expanded = f_b.unsqueeze(0);  

    auto l2_norms_f = torch::norm(f_a_expanded - f_b_expanded, 2, 2); 
    auto l2_norms_o = torch::norm(a.unsqueeze(1) - b.unsqueeze(0), 2, 2); 

    auto log_l2_norms_f = torch::log(l2_norms_f + 1e-8); 
    auto log_l2_norms_o = torch::log(l2_norms_o + 1e-8);  

    auto abs_loss = torch::mean(torch::pow(l2_norms_f - l2_norms_o, 2));
    auto log_loss = torch::mean(torch::pow(log_l2_norms_f - log_l2_norms_o, 2));

    auto loss = (*ratio) * log_loss + (1.0 - (*ratio)) * abs_loss;

    return loss;
}
torch::Tensor log_self_loss(
    shared_ptr<DES_MLP> model, 
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device
)
{
    model->to((*device));
    auto a = (*data).index({(*base)}).clone().to((*device));  
    auto b = (*data).index({(*cmp).reshape({-1})}).clone().to((*device));  
    // cout<<a.sizes()<<endl;
    // cout<<b.sizes()<<endl;
    b = b.reshape({(*cmp).sizes()[0],(*cmp).sizes()[1],-1});
    // cout<<b.sizes()<<endl;
    auto f_a = model->forward(a); 
    auto f_b = model->forward(b);
 
    auto f_a_expanded = f_a.unsqueeze(1);
    auto l2_norms_f = torch::norm(f_a_expanded - f_b, 2, 2); 
    auto l2_norms_o = torch::norm(a.unsqueeze(1) - b, 2, 2); 

    auto log_l2_norms_f = torch::log(l2_norms_f + 1e-8);  
    auto log_l2_norms_o = torch::log(l2_norms_o + 1e-8);  

    auto loss = torch::mean(torch::pow(log_l2_norms_f - log_l2_norms_o, 2));
    return loss;
}
torch::Tensor abs_self_loss(
    shared_ptr<DES_MLP> model, 
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device
)
{
    model->to((*device));
    auto a = (*data).index({(*base)}).clone().to((*device));
    auto b = (*data).index({(*cmp).reshape({-1})}).clone().to((*device));  
    // cout<<a.sizes()<<endl;
    // cout<<b.sizes()<<endl;
    b = b.reshape({(*cmp).sizes()[0],(*cmp).sizes()[1],-1}); 
    // cout<<b.sizes()<<endl;
    auto f_a = model->forward(a); 
    auto f_b = model->forward(b);  
 
    auto f_a_expanded = f_a.unsqueeze(1);  

    auto l2_norms_f = torch::norm(f_a_expanded - f_b, 2, 2); 
    auto l2_norms_o = torch::norm(a.unsqueeze(1) - b, 2, 2);

    // auto log_l2_norms_f = torch::log(l2_norms_f + 1e-8);  
    // auto log_l2_norms_o = torch::log(l2_norms_o + 1e-8);  

    auto loss = torch::mean(torch::pow(l2_norms_f - l2_norms_o, 2));
    return loss;
}