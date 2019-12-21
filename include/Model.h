

#ifndef CAPSTONE_MODEL_H
#define CAPSTONE_MODEL_H


#include "../include/Architecture.h"
#include <memory>
#include <string>

class Model {
private:
    std::shared_ptr<Architecture> _modelArchitecture;
    torch::optim::Adam _optimizer;
    torch::Device _device;
    bool _modelTrained = false;

public:
    Model();
    void train(int64_t epochs=10, int batchSize = 64, std::string dataDir = "./mnist_data", int workers = 2);
    void test(int batchSize=64, std::string dataDir="./mnist_data", int workers=2);
    void saveModel(std::string filename="model.pt");
    void loadModel(std::string filename = "model.pt");
    bool isModelTrained() const { return _modelTrained; }

};


#endif //CAPSTONE_MODEL_H
