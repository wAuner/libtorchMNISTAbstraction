

#ifndef CAPSTONE_MODEL_H
#define CAPSTONE_MODEL_H


#include "Architecture.h"
#include <memory>
#include <string>

class Model {
private:
    std::shared_ptr<Architecture> modelArchitecture;
    torch::optim::Adam optimizer;
    torch::Device device;

public:
    Model();
    void train(int64_t epochs=10, int batchSize = 64, std::string dataDir = "./mnist_data", int workers = 2);
    void test(int batchSize=64, std::string dataDir="./mnist_data", int workers=2);
    void saveModel(std::string filename="model.pt");
    void loadModel(std::string filename = "model.pt");

};


#endif //CAPSTONE_MODEL_H
