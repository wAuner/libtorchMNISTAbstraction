

#include "../include/Model.h"

#include <iostream>

void Model::train(int64_t epochs, int batchSize, std::string dataDir, int workers) {
    // create dataset and dataloader
    auto dataset = torch::data::datasets::MNIST(dataDir, torch::data::datasets::MNIST::Mode::kTrain)
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());
    const int dsSize = dataset.size().value();

    auto dataloader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions()
            .batch_size(batchSize).workers(workers));

    std::cout << "Start training..." << std::endl;
    for (int64_t epoch = 0; epoch < epochs; epoch++) {
        //double avgAccuracy = 0.0;
        int correct = 0;
        int64_t batchIndex = 0;
        for (torch::data::Example<>& batch : *dataloader) {
            torch::Tensor x_train = batch.data.to(_device);
            torch::Tensor y_train = batch.target.to(_device);
            _optimizer.zero_grad();

            torch::Tensor output = _modelArchitecture->forward(x_train);
            torch::Tensor loss = torch::nll_loss(output, y_train);
            torch::Tensor prediction = output.argmax(1);
            correct += prediction.eq(y_train).sum().item<int>();

            loss.backward();
            _optimizer.step();

            if (batchIndex % 100 == 0) {
                std::cout << "\rTraining epoch: " << epoch << "\tcurrent training loss: " << loss.item<float>()<< std::flush;

            }
        }
        std::cout << "\nAverage accuracy for epoch " << epoch << ": " << static_cast<double>(correct) / dsSize << std::endl;

    }
    std::cout << std::endl;
    _modelTrained = true;

}

void Model::test(int batchSize, std::string dataDir, int workers) {
    // create dataset and dataloader
    auto dataset = torch::data::datasets::MNIST(dataDir, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());
    const int dsSize = dataset.size().value();

    auto dataloader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions()
            .batch_size(batchSize).workers(workers));

    double accuracy = 0.0;
    int correct = 0;
    std::cout << "Start testing on test set..." << std::endl;
    for (torch::data::Example<>& batch : *dataloader) {
        torch::Tensor x_test = batch.data.to(_device);
        torch::Tensor y_test = batch.target.to(_device);

        torch::Tensor output = _modelArchitecture->forward(x_test);
        torch::Tensor predictions = output.argmax(1);
        correct += predictions.eq(y_test).sum().item<int>();
    }
    accuracy = static_cast<double>(correct) / dsSize;
    std::cout << "Model accuracy on test set: " << accuracy << std::endl;
}

// constructor
Model::Model() : _modelArchitecture(std::make_shared<Architecture>()),
                 _optimizer(_modelArchitecture->parameters(), torch::optim::AdamOptions(1e-5)),
                 _device(torch::kCPU) {
    if (torch::cuda::is_available()) {
        std::cout << "CUDA GPU is available, switching to GPU mode." << std::endl;
        _device = torch::kCUDA;
        _modelArchitecture->to(_device);
    }
}

void Model::saveModel(std::string filename) {
    std::cout << "Saving model..." << std::endl;
    torch::save(_modelArchitecture, filename);
}

void Model::loadModel(std::string filename) {
    std::cout << "Loading model..." << std::endl;
    torch::load(_modelArchitecture, filename);
    std::cout << "Loaded model successful." << std::endl;
    _modelTrained = true;
}

