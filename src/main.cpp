#include "../include/Model.h"

#include <algorithm>
#include <iostream>
#include <string>


int main() {
    Model mymodel;

    std::string command;
    do {
        std::cout << "What do you want to do? \nType one of the following: train/test/load/exit:";
        std::cin >> command;
        std::transform(command.begin(), command.end(), command.begin(), ::tolower);
        if (command == "train") {
            int64_t epochs = 0;
            std::cout << "How many epochs would you like to train?" << std::endl;
            std::cin >> epochs;
            mymodel.train(epochs);
            std::cout << "Would you like to save the model? If so, enter filename, else leave empty." << std::endl;
            std::string filename;
            std::cin >> filename;
            if (!filename.empty()) mymodel.saveModel(filename);
        } else if (command == "test") {
            if (mymodel.isModelTrained()) {
                mymodel.test();
            } else {
                std::cout << "Model has not been trained yet. Load model or train first." << std::endl;
            }
        } else if (command == "load") {
            std::string modelFile;
            std::cout << "Enter model name or load default model from ./model.pt" << std::endl;
            std::cin >> modelFile;
            mymodel.loadModel(modelFile);
        } else if (command != "exit") {
            std::cout << "Command not valid. Try again." << std::endl;
        }
    } while (command != "exit");



    return 0;
}
