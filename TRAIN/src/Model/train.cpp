#include "Model/train.h"

void single_train(int epoch,
            size_t cmp_cnt,
            shared_ptr<DES_MLP> &model,
            torch::Tensor &data, 
            KnnDataset &trainset,
            torch::Device &device)
{
    vector<int> base, cmp;
    double lr = 0.0001;
    auto tist=steady_clock::now(),tied=steady_clock::now();
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(lr));
    model->train();
    for(int ep=0;ep<epoch;ep++)
    {
        tist=steady_clock::now();
        trainset.shuffle_mp();
        int cnt = 0;
        double avg_loss = 0;
        for(size_t i=0;i<trainset.size();i++)
        {
            auto [id,tmp]=trainset.get(i);
            if((base.size()+1)*(cmp.size()+tmp.size())>cmp_cnt)
            {
                torch::Tensor base_tr = torch::tensor(base, torch::kInt32);
                torch::Tensor cmp_tr = torch::tensor(cmp, torch::kInt32);
                model->zero_grad();
                auto loss = log_contrastive_loss(model, &base_tr, &cmp_tr, &data, &device);
                loss.backward();
                optimizer.step();
                base.resize(0);
                cmp.resize(0);
                avg_loss += loss.item<double>();
                cnt++;
            }
            base.push_back(id);
            for(int tp:tmp) cmp.push_back(tp);
        }
        if(base.size())
        {
            torch::Tensor base_tr = torch::tensor(base, torch::kInt32);
            torch::Tensor cmp_tr = torch::tensor(cmp, torch::kInt32);
            model->zero_grad();
            auto loss = log_contrastive_loss(model, &base_tr, &cmp_tr, &data, &device);
            loss.backward();
            optimizer.step();
            avg_loss += loss.item<double>();
            cnt++;
        }
        tied=steady_clock::now();
        avg_loss /= cnt;
        cout << "time use:" << duration_cast<seconds>(tied - tist).count() << "s" << endl;
        cout << "Epoch " << ep << ", Loss: " << avg_loss << endl;
    }
}

void parallel_train_function_contrastive_loss(
    shared_ptr<DES_MLP> model,
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device,
    double *loss_val,
    float *ratio)
{
    model->zero_grad();
    auto loss = log_abs_contrastive_loss(model, base, cmp, data, device,ratio);
    *loss_val = loss.item<double>();
    loss.backward();
}

void parallel_train_function_self_loss(
    shared_ptr<DES_MLP> model,
    torch::Tensor *base,
    torch::Tensor *cmp,
    torch::Tensor *data, 
    torch::Device *device,
    double *loss_val)
{
    model->zero_grad();
    auto loss = abs_contrastive_loss(model, base, cmp, data, device);
    *loss_val = loss.item<double>();
    loss.backward();
}

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
)
{
    vector<torch::Device> devices;
    for(size_t i=0;i<device_id.size();i++)
    {
        devices.push_back(torch::Device(torch::kCUDA, device_id[i]));
    }
    model->to(devices[0]);
    size_t single_cmp_cnt=cmp_cnt/devices.size();
    double lr = 0.0001;
    auto tist=steady_clock::now(),tied=steady_clock::now();
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(lr));
    model->train();

    for(int ep=0;ep<epoch;ep++)
    {
        cout<<"epoch:"<<ep<<endl;
        vector<int> base,cmp;
        vector<torch::Tensor> base_tr(devices.size());
        vector<torch::Tensor> cmp_tr(devices.size());
        tist=steady_clock::now();
        trainset.shuffle_mp();
        int cnt = 0;
        double avg_loss = 0;
        for(size_t i=0;i<trainset.size();i++)
        {
            auto [id,tmp]=trainset.get(i);
            if((base.size()+1)*(cmp.size()+tmp.size())>cmp_cnt)
            {
                for(size_t j=0;j<devices.size();j++)
                    base_tr[j] = torch::tensor(base, torch::kInt32);
                vector<int> tmp_cmp;
                for(size_t j=0;j<cmp.size();j++)
                {
                    tmp_cmp.push_back(cmp[j]);
                    if((j+1)%(cmp.size()/devices.size())==0)
                    {
                        cmp_tr[(j+1)/(cmp.size()/devices.size())-1]=torch::tensor(tmp_cmp, torch::kInt32);
                        tmp_cmp.resize(0);
                    }
                }
                if(tmp_cmp.size())
                {
                    cmp_tr[cmp_tr.size()-1]=torch::tensor(tmp_cmp, torch::kInt32);
                    tmp_cmp.resize(0);
                }
                base.resize(0);
                cmp.resize(0);
                vector<double> losses(device_id.size(),0);
                model->zero_grad();
                {
                    double tp_double;
                    torch::Tensor a=torch::tensor({0}, torch::kInt32);
                    torch::Tensor b=torch::tensor({1}, torch::kInt32);
                    parallel_train_function_contrastive_loss(model, &a, &b, &data, &devices[0], &tp_double,&ratio);
                }    
                torch::save(model,"./tmp.tp");
                vector<shared_ptr<DES_MLP>> models(device_id.size(),nullptr);
                for(size_t j=0;j<device_id.size();j++)
                {
                    models[j]=make_shared<DES_MLP>(in_dim,out_dim);
                    torch::load(models[j],"./tmp.tp");
                }
                vector<thread*> thread_vector;
                for(size_t j=0;j<device_id.size();j++)
                {
                    thread_vector.push_back(new thread(parallel_train_function_contrastive_loss, models[j],
                        &base_tr[j], &cmp_tr[j], &data, &devices[j], &losses[j] ,&ratio));
                }
                torch::Tensor fc1_w,fc2_w,fc3_w,fc4_w,fc5_w;
                torch::Tensor fc1_b,fc2_b,fc3_b,fc4_b,fc5_b;
                for(size_t j=0;j<device_id.size();j++)
                {
                    thread_vector[j]->join();
                    avg_loss += losses[j];
                    cnt++;
                    if(j==0)
                    {
                        fc1_w = torch::zeros_like(models[j]->fc1->weight.grad().clone());
                        fc2_w = torch::zeros_like(models[j]->fc2->weight.grad().clone());
                        fc3_w = torch::zeros_like(models[j]->fc3->weight.grad().clone());
                        fc4_w = torch::zeros_like(models[j]->fc4->weight.grad().clone());
                        fc5_w = torch::zeros_like(models[j]->fc5->weight.grad().clone());
                        fc1_b = torch::zeros_like(models[j]->fc1->bias.grad().clone());
                        fc2_b = torch::zeros_like(models[j]->fc2->bias.grad().clone());
                        fc3_b = torch::zeros_like(models[j]->fc3->bias.grad().clone());
                        fc4_b = torch::zeros_like(models[j]->fc4->bias.grad().clone());
                        fc5_b = torch::zeros_like(models[j]->fc5->bias.grad().clone());
                    }
                    fc1_w += models[j]->fc1->weight.grad().clone().to(devices[0]);
                    fc2_w += models[j]->fc2->weight.grad().clone().to(devices[0]);
                    fc3_w += models[j]->fc3->weight.grad().clone().to(devices[0]);
                    fc4_w += models[j]->fc4->weight.grad().clone().to(devices[0]);
                    fc5_w += models[j]->fc5->weight.grad().clone().to(devices[0]);
                    fc1_b += models[j]->fc1->bias.grad().clone().to(devices[0]);
                    fc2_b += models[j]->fc2->bias.grad().clone().to(devices[0]);
                    fc3_b += models[j]->fc3->bias.grad().clone().to(devices[0]);
                    fc4_b += models[j]->fc4->bias.grad().clone().to(devices[0]);
                    fc5_b += models[j]->fc5->bias.grad().clone().to(devices[0]);
                }
                fc1_w /= device_id.size();
                fc2_w /= device_id.size();
                fc3_w /= device_id.size();
                fc4_w /= device_id.size();
                fc5_w /= device_id.size();
                fc1_b /= device_id.size();
                fc2_b /= device_id.size();
                fc3_b /= device_id.size();
                fc4_b /= device_id.size();
                fc5_b /= device_id.size();
                model->fc1->weight.grad().copy_(fc1_w);
                model->fc2->weight.grad().copy_(fc2_w);
                model->fc3->weight.grad().copy_(fc3_w);
                model->fc4->weight.grad().copy_(fc4_w);
                model->fc5->weight.grad().copy_(fc5_w);
                model->fc1->bias.grad().copy_(fc1_b);
                model->fc2->bias.grad().copy_(fc2_b);
                model->fc3->bias.grad().copy_(fc3_b);
                model->fc4->bias.grad().copy_(fc4_b);
                model->fc5->bias.grad().copy_(fc5_b);
                optimizer.step();
            }
            base.push_back(id);
            for(int tp:tmp) cmp.push_back(tp);
        }
        
        if(cmp.size()>=devices.size())
        {
            for(size_t j=0;j<devices.size();j++)
                base_tr[j] = torch::tensor(base, torch::kInt32);
            vector<int> tmp_cmp;
            for(size_t j=0;j<cmp.size();j++)
            {
                tmp_cmp.push_back(cmp[j]);
                if((j+1)%(cmp.size()/devices.size())==0)
                {
                    cmp_tr[(j+1)/(cmp.size()/devices.size())-1]=torch::tensor(tmp_cmp, torch::kInt32);
                    tmp_cmp.resize(0);
                }
            }
            if(tmp_cmp.size())
            {
                cmp_tr[cmp_tr.size()-1]=torch::tensor(tmp_cmp, torch::kInt32);
                tmp_cmp.resize(0);
            }
            base.resize(0);
            cmp.resize(0);
            vector<double> losses(device_id.size(),0);
            model->zero_grad();
            {
                double tp_double;
                torch::Tensor a=torch::tensor({0}, torch::kInt32);
                torch::Tensor b=torch::tensor({1}, torch::kInt32);
                parallel_train_function_contrastive_loss(model, &a, &b, &data, &devices[0], &tp_double, &ratio);
            }    
            torch::save(model,"./tmp.tp");
            vector<shared_ptr<DES_MLP>> models(device_id.size(),nullptr);
            for(size_t j=0;j<device_id.size();j++)
            {
                models[j]=make_shared<DES_MLP>(in_dim,out_dim);
                torch::load(models[j],"./tmp.tp");
            }
            vector<thread*> thread_vector;
            for(size_t j=0;j<device_id.size();j++)
            {
                thread_vector.push_back(new thread(parallel_train_function_contrastive_loss, models[j],
                    &base_tr[j], &cmp_tr[j], &data, &devices[j], &losses[j], &ratio));
            }
            torch::Tensor fc1_w,fc2_w,fc3_w,fc4_w,fc5_w;
            torch::Tensor fc1_b,fc2_b,fc3_b,fc4_b,fc5_b;
            for(size_t j=0;j<device_id.size();j++)
            {
                thread_vector[j]->join();
                avg_loss += losses[j];
                cnt++;
                if(j==0)
                {
                    fc1_w = torch::zeros_like(models[j]->fc1->weight.grad().clone());
                    fc2_w = torch::zeros_like(models[j]->fc2->weight.grad().clone());
                    fc3_w = torch::zeros_like(models[j]->fc3->weight.grad().clone());
                    fc4_w = torch::zeros_like(models[j]->fc4->weight.grad().clone());
                    fc5_w = torch::zeros_like(models[j]->fc5->weight.grad().clone());
                    fc1_b = torch::zeros_like(models[j]->fc1->bias.grad().clone());
                    fc2_b = torch::zeros_like(models[j]->fc2->bias.grad().clone());
                    fc3_b = torch::zeros_like(models[j]->fc3->bias.grad().clone());
                    fc4_b = torch::zeros_like(models[j]->fc4->bias.grad().clone());
                    fc5_b = torch::zeros_like(models[j]->fc5->bias.grad().clone());
                }
                fc1_w += models[j]->fc1->weight.grad().clone().to(devices[0]);
                fc2_w += models[j]->fc2->weight.grad().clone().to(devices[0]);
                fc3_w += models[j]->fc3->weight.grad().clone().to(devices[0]);
                fc4_w += models[j]->fc4->weight.grad().clone().to(devices[0]);
                fc5_w += models[j]->fc5->weight.grad().clone().to(devices[0]);
                fc1_b += models[j]->fc1->bias.grad().clone().to(devices[0]);
                fc2_b += models[j]->fc2->bias.grad().clone().to(devices[0]);
                fc3_b += models[j]->fc3->bias.grad().clone().to(devices[0]);
                fc4_b += models[j]->fc4->bias.grad().clone().to(devices[0]);
                fc5_b += models[j]->fc5->bias.grad().clone().to(devices[0]);
            }
            fc1_w /= device_id.size();
            fc2_w /= device_id.size();
            fc3_w /= device_id.size();
            fc4_w /= device_id.size();
            fc5_w /= device_id.size();
            fc1_b /= device_id.size();
            fc2_b /= device_id.size();
            fc3_b /= device_id.size();
            fc4_b /= device_id.size();
            fc5_b /= device_id.size();
            model->fc1->weight.grad().copy_(fc1_w);
            model->fc2->weight.grad().copy_(fc2_w);
            model->fc3->weight.grad().copy_(fc3_w);
            model->fc4->weight.grad().copy_(fc4_w);
            model->fc5->weight.grad().copy_(fc5_w);
            model->fc1->bias.grad().copy_(fc1_b);
            model->fc2->bias.grad().copy_(fc2_b);
            model->fc3->bias.grad().copy_(fc3_b);
            model->fc4->bias.grad().copy_(fc4_b);
            model->fc5->bias.grad().copy_(fc5_b);
            optimizer.step();
        }
        tied=steady_clock::now();
        avg_loss /= cnt;
        cout << "time use:" << duration_cast<seconds>(tied - tist).count() << "s" << endl;
        cout << "Epoch " << ep << ", Loss: " << avg_loss << endl;
        // if((ep+1)%20==0)
        model->to(torch::kCPU);
        torch::save(model, save_path + ".cpt");
        model->to(devices[0]);
    }
}
