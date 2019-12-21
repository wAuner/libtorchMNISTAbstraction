

#ifndef CAPSTONE_ARCHITECTURE_H
#define CAPSTONE_ARCHITECTURE_H

#include <torch/torch.h>

struct Architecture : torch::nn::Module {
    // constructor
    Architecture();
    // calculate forward pass
    torch::Tensor forward(torch::Tensor);

private:
    torch::nn::Conv2d convLayer2;
    torch::nn::Conv2d convLayer3;
    torch::nn::Conv2d convLayer1;
    torch::nn::Conv2d convLayer4;
    torch::nn::Linear finalLayer;
};


#endif //CAPSTONE_ARCHITECTURE_H
