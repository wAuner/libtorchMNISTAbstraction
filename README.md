# CppND Capstone: highlevel libtorch MNIST API

The project uses the Pytorch C++ API libtorch to provide a higher level API to train, test, load and save an image
classification model for the MNIST dataset. This higher level API creates a model that exemplary abstracts the lower level
tasks for a specific use case. Supports GPU and CPU training / inference.
There is also an interactive commandline app that lets you control these actions via terminal.

## Dependencies / prerequisites
* [libtorch](https://pytorch.org/get-started/locally/)
   * prebuild binary download available. just download and use the `CMAKE_PREFIX_PATH`option
   * GPU or CPU, for GPU Cuda 10.1 and cudnn is required
* a compiler with OpenMP like g++ (clang on mac does not come with OpenMP)
* the mnist dataset can be downloaded using the script. `python mnist_download.py`
    * the dataset **must** be located or linked to the same directory as the binary e.g. execute the download command from `build`
    
## Build instructions
From the project directory:
* `mkdir build && cd build`
* prepare dataset:
   * `mkdir mnist_data && cd mnist_data`
   * download and extract dataset `python ../../mnist_download.py`
   * return to build dir `cd ..`
The project structure should now look like this:
```
.
|-- build
|   `-- mnist_data
|       |-- t10k-images-idx3-ubyte
|       |-- t10k-images-idx3-ubyte.gz
|       |-- t10k-labels-idx1-ubyte
|       |-- t10k-labels-idx1-ubyte.gz
|       |-- train-images-idx3-ubyte
|       |-- train-images-idx3-ubyte.gz
|       |-- train-labels-idx1-ubyte
|       `-- train-labels-idx1-ubyte.gz
|-- CMakeLists.txt
|-- include
|   |-- Architecture.h
|   `-- Model.h
|-- mnist_download.py
|-- README.md
`-- src
    |-- Architecture.cpp
    |-- main.cpp
    `-- Model.cpp
```
  
* to build the project from build dir: 
   * `cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..` followed by 
   * `make`
* execute `./Capstone`
    
## Code structure
```
src/
|-- Architecture.cpp    defines the structure of the underlying neural network. the architecture is a member of the Model class
|-- main.cpp            contains the interactive command line control flow
`-- Model.cpp           implements the higher level Model api and its functions train/test/load/save
```

## Examples of addressed rubric points
### Loops, Functions, I/O
| Rubric  | example in project |
| ------------- | ------------- |
| The project demonstrates an understanding of C++ functions and control structures.  | control flow structures and loops are used, member functions are defined and implemented  |
| The project reads data from a file and process the data, or the program writes data to a file.  | the model class (load/save functions) is able to read and write data to/from disk  |
| The project accepts user input and processes the input. | main.cpp implements an interactive control flow for the command line |


### Object Oriented Programming
| Rubric  | example in project |
| ------------- | ------------- |
| The project uses Object Oriented Programming techniques.  | Model and Architecture class defined and implemented  |
| Classes use appropriate access specifiers for class members. | All class data members are explicitly specified as public or private.|
| Class constructors utilize member initialization lists. | Constructor of Model class uses initialization lists for all members |
| Classes abstract implementation details from their interfaces. | The model class is an abstraction for the underlying torch implementation |
 
and many more...
