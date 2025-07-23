#include <iostream>
#include <chrono>
#include "CPPNN/VulkanGPUCompute.h"
#include "CPPNN/ModelDevice.h"
#include "CPPNN/Dataset.h"

#define EIGEN_DONT_PARALLELIZE 0
#define EIGEN_USE_THREADS

static int s_AllocationCount = 0;

void *operator new(size_t size)
{
    ++s_AllocationCount;
    return malloc(size);
}
/*

void conv_example()
{
    using namespace NN;

    auto conv = std::make_shared<Conv2DLayer>(1, 2);
    conv->connect(1);

    conv->filters(0, 0, 0, 0) = 1.0;
    conv->filters(0, 0, 0, 1) = 0.0;
    conv->filters(0, 0, 1, 0) = 0.0;
    conv->filters(0, 0, 1, 1) = -1.0;

    conv->biases(0, 0) = 0.0;

    // Entrada: [1 canal, 3 alto, 3 ancho]
    Tensor input(1, 3, 3);
    input(0, 0, 0) = 1.0;
    input(0, 0, 1) = 2.0;
    input(0, 0, 2) = 3.0;
    input(0, 1, 0) = 4.0;
    input(0, 1, 1) = 5.0;
    input(0, 1, 2) = 6.0;
    input(0, 2, 0) = 7.0;
    input(0, 2, 1) = 8.0;
    input(0, 2, 2) = 9.0;

    Tensor output;
    conv->forward(input, output);

    std::cout << "Resultado de conv_example():\n";
    output.print(); // Esperado: [-4, -4], [-4, -4]
}

void conv_example_larger()
{
    using namespace NN;

    Tensor input(1, 28, 28); // Imagen de entrada de 28x28 y 1 canal
    input.fill(0.0);         // Llenamos con ceros, puede ser con datos aleatorios

    // Capa Conv2D #1: 32 filtros de 3x3
    Conv2DLayer conv1(32, 3);
    conv1.connect(1); // Conectamos 1 canal de entrada
    Tensor output1;
    conv1.forward(input, output1);
    std::cout << "Capa Conv2D #1 (32 filtros 3x3) - Shape: (" << output1.getDim(0) << ", " << output1.getDim(1) << ", " << output1.getDim(2) << ")\n";

    // Capa MaxPooling: 2x2
    MaxPooling2D pool1(2);
    Tensor output2;
    pool1.forward(output1, output2);
    std::cout << "MaxPooling #1 (2x2) - Shape: (" << output2.getDim(0) << ", " << output2.getDim(1) << ", " << output2.getDim(2) << ")\n";

    // Capa Conv2D #2: 64 filtros de 3x3
    Conv2DLayer conv2(64, 3);
    conv2.connect(32); // 32 canales de entrada
    Tensor output3;
    conv2.forward(output2, output3);
    std::cout << "Capa Conv2D #2 (64 filtros 3x3) - Shape: (" << output3.getDim(0) << ", " << output3.getDim(1) << ", " << output3.getDim(2) << ")\n";

    // Capa MaxPooling: 2x2
    MaxPooling2D pool2(2);
    Tensor output4;
    pool2.forward(output3, output4);
    std::cout << "MaxPooling #2 (2x2) - Shape: (" << output4.getDim(0) << ", " << output4.getDim(1) << ", " << output4.getDim(2) << ")\n";

    // Capa Conv2D #3: 64 filtros de 3x3
    Conv2DLayer conv3(64, 3);
    conv3.connect(64); // 64 canales de entrada
    Tensor output5;
    conv3.forward(output4, output5);
    std::cout << "Capa Conv2D #3 (64 filtros 3x3) - Shape: (" << output5.getDim(0) << ", " << output5.getDim(1) << ", " << output5.getDim(2) << ")\n";

    // Capa Flatten
    Flatten flatten;
    Tensor output6;
    flatten.forward(output5, output6);
    std::cout << "Flatten - Shape: (" << output6.getDim(0) << ")\n";

    // Capa Densa: 64 neuronas
    DenseLayer dense(64);
    dense.connect(576);
    Tensor output7;
    dense.forward(output6, output7);
    std::cout << "Capa Densa (64 neuronas) - Shape: (" << output7.getDim(0) << ")\n";

    // Capa de salida: 10 neuronas
    DenseLayer dense_out(10);
    dense_out.connect(64);
    Tensor output8;
    dense_out.forward(output7, output8);
    std::cout << "Capa de salida (10 neuronas) - Shape: (" << output8.getDim(0) << ")\n";
}

void conv_example_larger2()
{
    using namespace NN;

    Tensor input(1, 7, 7);
    input.fill(1.0);

    Conv2DLayer conv1(2, 3);
    conv1.connect(1);
    Tensor output1;
    conv1.forward(input, output1);
    std::cout << "Capa Conv2D #1 (2 filtros 3x3) - Shape: ("
              << output1.getDim(0) << ", " << output1.getDim(1) << ", "
              << output1.getDim(2) << ")\n";

    MaxPooling2D pool1(2);
    Tensor output2;
    pool1.forward(output1, output2);
    std::cout << "MaxPooling #1 (2x2) - Shape: ("
              << output2.getDim(0) << ", " << output2.getDim(1) << ", "
              << output2.getDim(2) << ")\n";

    Flatten flatten;
    Tensor output5;
    flatten.forward(output2, output5);
    std::cout << "Flatten - Shape: (" << output5.getDim(0) << ")\n";

    DenseLayer dense(4);
    dense.connect(8); // Tamaño de la entrada de la capa densa (2x2 = 4)
    Tensor output6;
    dense.forward(output5, output6);
    std::cout << "Capa Densa (4 neuronas) - Shape: (" << output6.getDim(0) << ")\n";

    // 8. Capa de salida: 2 neuronas
    DenseLayer dense_out(2);
    dense_out.connect(4); // Conectamos con 4 neuronas de entrada
    Tensor output7;
    dense_out.forward(output6, output7);
    std::cout << "Capa de salida (2 neuronas) - Shape: (" << output7.getDim(0) << ")\n";
}

*/
#include <cmath>

uint16_t floatToHalf(float f)
{
    uint32_t *f_bits = reinterpret_cast<uint32_t *>(&f);

    // Extraer el signo (1 bit), exponente (5 bits) y fracción (10 bits)
    uint32_t sign = (*f_bits >> 31) & 0x1;
    uint32_t exponent = (*f_bits >> 23) & 0xFF;
    uint32_t fraction = (*f_bits) & 0x7FFFFF;

    uint16_t half_sign = sign;
    int16_t half_exponent = exponent - 127 + 15; // Ajuste de exponente

    uint16_t half_fraction = 0;
    if (exponent > 0 && exponent < 255)
    { // No es un número subnormal
        half_fraction = fraction >> 13;
    }

    // Casos especiales: cero, infinito y NaN
    if (exponent == 0)
    {
        // Subnormal
        half_exponent = 0;
        half_fraction = fraction >> 13;
    }
    if (exponent == 255)
    {
        // Infinito o NaN
        if (fraction == 0)
        {
            return (half_sign << 15) | (0x1F << 10); // Infinito
        }
        return (half_sign << 15) | (0x1F << 10) | (fraction >> 13); // NaN
    }

    // Control de saturación si el exponente es mayor a 30 (infinito)
    if (half_exponent > 30)
    {
        half_exponent = 0x1F; // Infinito
        half_fraction = 0;
    }

    // Combinamos todos los componentes
    uint16_t half = (half_sign << 15) | (half_exponent << 10) | half_fraction;
    return half;
}
float halfToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t fraction = h & 0x3FF;

    if (exponent == 0)
    { // subnormal
        return sign ? -0.0f : 0.0f;
    }
    if (exponent == 31)
    { // infinito o NaN
        return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }

    // Calcular el valor en float
    float result = std::ldexp(fraction, -10);
    result += 1.0f;                            // La fracción está implícita como 1
    result *= std::ldexp(1.0f, exponent - 15); // Ajuste de exponente

    return sign ? -result : result;
}

void vulkantest()
{
    VkBuffer buffer;
}

#include <Eigen/Dense>
#include <thread>

int main()
{
    using namespace NN;
    Eigen::setNbThreads(std::thread::hardware_concurrency());
    srand(static_cast<unsigned int>(time(0)));

    // VulkanGPUCompute vulkan_compute;
    int epochs = 1000;

    Dataset dataset("Crop_recommendation.csv", 7);
    auto [train_ds, test_ds] = dataset.split(0.2);

    Tensor<NN::CPU> X_train = train_ds.features_tensor();
    Tensor<NN::CPU> y_train = train_ds.labels_tensor();
    Tensor<NN::CPU> y_trainOneHOT(y_train.getDim(0), 22);
    Tensor<NN::CPU> X_test = test_ds.features_tensor();
    Tensor<NN::CPU> y_test = test_ds.labels_tensor();
    Tensor<NN::CPU> y_testOneHOT(y_test.getDim(0), 22);

    y_trainOneHOT.fill(0.0f);
    for (int i = 0; i < y_train.getDim(0); ++i)
        y_trainOneHOT(i, y_train(i)) = 1.0f;

    y_testOneHOT.fill(0.0f);
    for (int i = 0; i < y_test.getDim(0); ++i)
        y_testOneHOT(i, y_test(i)) = 1.0f;

    std::cout << "X_train: " << X_train.getNdim() << ", " << X_train.getSize() << std::endl;
    std::cout << "y_train: " << y_train.getNdim() << ", " << y_train.getSize() << std::endl;
    std::cout << "X_test: " << X_test.getNdim() << ", " << X_test.getSize() << std::endl;
    std::cout << "y_test: " << y_test.getNdim() << ", " << y_test.getSize() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    Model<NN::CPU> model({{Layer::Input, 7}, {Layer::Dense, 128, Activation::ReLU}, {Layer::Dense, 64, Activation::ReLU}, {Layer::Dense, 32, Activation::ReLU}, {Layer::Dense, 22, Activation::Softmax}}, NN::Loss::CrossEntropy, NN::Optimizer::Adam);
    model.enableMultiThreading();
    model.fit(X_train, y_trainOneHOT, 0.001, 40, 32);
    auto end = std::chrono::high_resolution_clock::now();
    double time_first = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: " << time_first << std::endl;

    int total_correct = 0;
    int total_samples = 0;
    Tensor input = X_test.createView(0, 2);
    Tensor target = y_test.createView(0, 2);
    Tensor<NN::CPU> output;

    model.predict(input, output);
    output.print();
    target.print();

    float lr = 0.4;

    Tensor<NN::CPU> A(2, 3);
    A(0, 0) = 1.0f;
    A(0, 1) = 2.0f;
    A(0, 2) = 3.0f;
    A(1, 0) = 4.0f;
    A(1, 1) = 5.0f;
    A(1, 2) = 6.0f;

    Tensor<NN::CPU> B(3, 2);
    B(0, 0) = 7.0f;
    B(0, 1) = 8.0f;
    B(1, 0) = 9.0f;
    B(1, 1) = 10.0f;
    B(2, 0) = 11.0f;
    B(2, 1) = 12.0f;

    Tensor<NN::CPU> C = B.transposeView(0, 1).matmul(A.transposeView(0, 1));
    Tensor<NN::CPU> CThreaded = B.transpose().matmulThreaded(A.transpose());
    C.print();
    CThreaded.print();
}