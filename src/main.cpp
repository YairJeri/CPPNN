#include <iostream>
#include "CPPNN/TensorDevice.h"

int main() {
    std::cout << "CPPNN Core initialized." << std::endl;
    NN::Tensor<NN::CPU> t(2, 4);
    t.setZero();
    std::cout << "Tensor shape: [" << t.getDim(0) << ", " << t.getDim(1) << "]" << std::endl;
    return 0;
}
