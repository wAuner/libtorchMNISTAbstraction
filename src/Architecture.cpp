

#include "../include/Architecture.h"

Architecture::Architecture() : convLayer1(torch::nn::Conv2dOptions(1, 16, 3)),
                               convLayer2(torch::nn::Conv2dOptions(16, 32, 3)),
                               convLayer3(torch::nn::Conv2dOptions(32, 64, 3)),
                               convLayer4(torch::nn::Conv2dOptions(64, 128, 3)),
                               finalLayer(51200, 10) {
    //std::cout << convLayer4. << std::endl;
    register_module("convLayer1", convLayer1);
    register_module("convLayer2", convLayer2);
    register_module("convLayer3", convLayer3);
    register_module("convLayer4", convLayer4);
    register_module("finalLayer", finalLayer);
}

torch::Tensor Architecture::forward(torch::Tensor x) {
    x = torch::relu(convLayer1(x));
    x = torch::relu(convLayer2(x));
    x = torch::relu(convLayer3(x));
    x = torch::relu(convLayer4(x));
    x = x.view({-1, 51200});
    x = finalLayer(x);
    return torch::log_softmax(x, 1);
}