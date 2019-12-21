# CppND Capstone: Interactive libtorch command line MNIST app

The project uses the Pytorch C++ API libtorch to provide a higher level API to train, test, load and save an image
classification model for the MNIST dataset. This higher level API creates a model that exemplary abstracts the lower level
tasks for a specific use case. Supports GPU and CPU training / inference.
There is also an interactive commandline app that lets you control these actions via terminal.

## Dependencies / prerequisites
* [libtorch](https://pytorch.org/get-started/locally/) 
    * GPU or CPU, for GPU Cuda 10.1 and cudnn is needed
* a compiler with OpenMP like g++ (clang on mac does not come with OpenMP)
* the mnist dataset can be downloaded using the script. `python mnist_download.py`
    * the dataset **must** be located or linked to the same directory as the binary e.g. execute the download command from `build`
    
## Code structure
```
src/
|-- Architecture.cpp    defines the structure of the underlying neural network. the architecture is a member of the Model class
|-- main.cpp            contains the interactive command line control flow
`-- Model.cpp           implements the higher level Model api and its functions train/test/load/save
```

## Addressed rubric points
### Loops, Functions, I/O
