#include "../include/Model.h"

#include <torch/torch.h>
#include <iostream>



int main() {
    Model mymodel;
    mymodel.train(5);
    mymodel.saveModel("./new_model.pt");
    mymodel.loadModel("./new_model.pt");
    mymodel.test();

    return 0;
}
