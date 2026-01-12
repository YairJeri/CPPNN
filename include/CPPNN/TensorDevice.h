#pragma once

#include "enums.h"
#include "TensorBase.h"
#include "VulkanStructs.h"
#include <cstring>
#include <cassert>
#include <memory>

namespace NN
{
    template <typename Device>
    class Tensor
    {
    };

    template <>
    class Tensor<NN::CPU> : public TensorBase
    {
    public:
        Tensor() : TensorBase() {}
        Tensor(int i) : TensorBase(i) {}
        Tensor(int i, int j) : TensorBase(i, j) {}
        Tensor(int i, int j, int k) : TensorBase(i, j, k) {}
        Tensor(int i, int j, int k, int l) : TensorBase(i, j, k, l) {}
        Tensor(int *shape, int ndim) : TensorBase(shape, ndim) {}
        Tensor(const Tensor &other);
        Tensor &operator=(const Tensor &other);
        Tensor &operator=(Tensor &&other) noexcept;

        void apply(float (*func)(float));
        Tensor createView(size_t start, size_t count) const;
        Tensor transposeView(int axis1, int axis2) const;
        Tensor transpose() const;
        Tensor matmul(const Tensor &other) const;
        Tensor matmulThreaded(const Tensor &other) const;

        Tensor sumAxis0() const
        {
            assert(ndim == 2);
            int rows = shape[0];
            int cols = shape[1];

            Tensor result(1, cols);
            for (int j = 0; j < cols; ++j)
                result(j) = 0.0f;

            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    result(j) += (*this)(i, j);
                }
            }

            return result;
        }

        static Tensor<NN::CPU> forward(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &weights, const Tensor<NN::CPU> &biases);

        Tensor &broadcastAdd(const Tensor &other);

        Tensor operator*(float scalar) const;
        Tensor &operator*=(float scalar);
        Tensor operator*(const Tensor &other) const;
        Tensor &operator*=(const Tensor &other);
        Tensor operator+(const Tensor &other) const;
        Tensor &operator+=(const Tensor &other);
        Tensor operator-(const Tensor &other) const;
        Tensor &operator-=(const Tensor &other);

        void print();

        float &operator()(int i);
        const float &operator()(int i) const;
        float &operator()(int i, int j);
        const float &operator()(int i, int j) const;
        float &operator()(int i, int j, int k);
        const float &operator()(int i, int j, int k) const;
        float &operator()(int i, int j, int k, int l);
        const float &operator()(int i, int j, int k, int l) const;
        /*
                Tensor(const Tensor &other) : TensorBase()
                {
                    ndim = other.ndim;
                    size = other.size;
                    own_data = true;

                    for (int i = 0; i < 4; ++i)
                    {
                        shape[i] = other.shape[i];
                        strds[i] = other.strds[i];
                    }

                    data = new float[size];
                    std::memcpy(data, other.data, size * sizeof(float));
                }

                void apply(float (*func)(float))
                {
                    for (int i = 0; i < size; ++i)
                        data[i] = func(data[i]);
                }

                Tensor createView(size_t start, size_t count) const
                {
                    if (start < 0 || count < 1 || start + count > shape[0])
                    {
                        throw std::out_of_range("View out of range");
                    }

                    Tensor view;
                    view.ndim = ndim;
                    view.size = (size / shape[0]) * count;

                    view.shape[0] = count;
                    for (int d = 1; d < ndim; ++d)
                    {
                        view.shape[d] = shape[d];
                    }

                    for (int d = 0; d < ndim; ++d)
                    {
                        view.strds[d] = strds[d];
                    }

                    size_t offset = start * strds[0];
                    view.data = data + offset;
                    view.own_data = false;

                    return view;
                }

                Tensor transposeView(int axis1, int axis2) const
                {
                    Tensor view;
                    view.ndim = ndim;
                    view.size = size;
                    for (int d = 0; d < ndim; ++d)
                    {
                        view.shape[d] = shape[d];
                        view.strds[d] = strds[d];
                    }

                    std::swap(view.shape[axis1], view.shape[axis2]);
                    std::swap(view.strds[axis1], view.strds[axis2]);

                    view.data = data;
                    view.own_data = false;

                    return view;
                }

                Tensor matmul(const Tensor &other) const
                {
                    if (ndim != 2 || other.ndim != 2)
                        throw std::invalid_argument("Tensors must be 2D");
                    if (shape[1] != other.shape[0])
                        throw std::invalid_argument("Dimension mismatch for matrix multiplication");
                    Tensor result(shape[0], other.shape[1]);
                    result.fill(0.0);
                    for (int i = 0; i < shape[0]; ++i)
                    {
                        for (int j = 0; j < other.shape[1]; ++j)
                        {
                            for (int k = 0; k < shape[1]; ++k)
                            {
                                result(i, j) += data[i * strds[0] + k * strds[1]] * other(k, j);
                            }
                        }
                    }
                    return result;
                }

                Tensor &operator=(const Tensor &other)
                {
                    if (this != &other)
                    {
                        if (size != other.size)
                        {
                            if (own_data && data)
                                delete[] data;
                            size = other.size;
                            data = new float[size];
                        }

                        ndim = other.ndim;
                        own_data = true;

                        for (int i = 0; i < 4; ++i)
                        {
                            shape[i] = other.shape[i];
                            strds[i] = other.strds[i];
                        }

                        std::memcpy(data, other.data, size * sizeof(float));
                    }
                    return *this;
                }
                Tensor &operator=(Tensor &&other) noexcept
                {
                    if (this != &other)
                    {
                        if (own_data && data)
                            delete[] data;

                        ndim = other.ndim;
                        size = other.size;
                        own_data = other.own_data;
                        data = other.data;

                        for (int i = 0; i < 4; ++i)
                        {
                            shape[i] = other.shape[i];
                            strds[i] = other.strds[i];
                        }

                        other.ndim = 0;
                        other.size = 0;
                        other.own_data = false;
                        other.data = nullptr;
                    }
                    return *this;
                }

                Tensor operator*(float scalar) const
                {
                    Tensor result(*this);
                    for (int i = 0; i < size; ++i)
                        result.data[i] = data[i] * scalar;
                    return result;
                }
                Tensor &operator*=(float scalar)
                {
                    for (int i = 0; i < size; ++i)
                        data[i] *= scalar;
                    return *this;
                }

                Tensor operator*(const Tensor &other) const
                {
                    if (ndim != other.ndim)
                        throw std::invalid_argument("*: Tensors must have same number of dimensions");

                    for (int i = 0; i < ndim; ++i)
                    {
                        if (shape[i] != other.shape[i])
                            throw std::invalid_argument("Tensors must have the same shape for element-wise multiplication");
                    }

                    Tensor result(*this);
                    for (int i = 0; i < size; ++i)
                    {
                        result.data[i] *= other.data[i];
                    }
                    return result;
                }

                Tensor &operator*=(const Tensor &other)
                {
                    if (ndim != other.ndim)
                        throw std::invalid_argument("*=: Tensors must have same number of dimensions");

                    for (int i = 0; i < ndim; ++i)
                    {
                        if (shape[i] != other.shape[i])
                            throw std::invalid_argument("Tensors must have the same shape for element-wise multiplication");
                    }

                    for (int i = 0; i < size; ++i)
                    {
                        data[i] *= other.data[i];
                    }
                    return *this;
                }

                Tensor operator+(const Tensor &other) const
                {
                    if (ndim != other.ndim)
                        throw std::invalid_argument("+: Tensors must have same number of dimensions");

                    for (int i = 0; i < ndim; ++i)
                    {
                        if (shape[i] != other.shape[i])
                            throw std::invalid_argument("Tensors must have the same shape for element-wise addition");
                    }

                    Tensor result(*this);
                    for (int i = 0; i < size; ++i)
                    {
                        result.data[i] += other.data[i];
                    }
                    return result;
                }

                Tensor &operator+=(const Tensor &other)
                {
                    if (ndim != other.ndim)
                        throw std::invalid_argument("+=: Tensors must have same number of dimensions");

                    for (int i = 0; i < ndim; ++i)
                    {
                        if (shape[i] != other.shape[i])
                            throw std::invalid_argument("Tensors must have the same shape for element-wise addition");
                    }

                    for (int i = 0; i < size; ++i)
                    {
                        data[i] += other.data[i];
                    }
                    return *this;
                }

                Tensor operator-(const Tensor &other) const
                {
                    if (shape[0] != other.shape[0] || shape[1] != other.shape[1] || shape[2] != other.shape[2] || shape[3] != other.shape[3])
                        throw std::invalid_argument("-: Tensors must have the same shape for element-wise subtraction");

                    Tensor result(*this);
                    for (int i = 0; i < size; ++i)
                    {
                        result.data[i] -= other.data[i];
                    }
                    return result;
                }

                Tensor &operator-=(const Tensor &other)
                {
                    if (shape[0] != other.shape[0] || shape[1] != other.shape[1] || shape[2] != other.shape[2] || shape[3] != other.shape[3])
                        throw std::invalid_argument("-=: Tensors must have the same shape for element-wise subtraction");

                    for (int i = 0; i < size; ++i)
                    {
                        data[i] -= other.data[i];
                    }
                    return *this;
                }

                void print()
                {
                    std::cout << "shp: [";
                    for (int i = 0; i < 4; ++i)
                    {
                        std::cout << shape[i];
                        if (i < 3)
                            std::cout << ", "; // Asegura que no haya coma después del último valor
                    }
                    std::cout << "], size: " << size << ", strds: [";
                    for (int i = 0; i < 4; ++i)
                    {
                        std::cout << strds[i];
                        if (i < 3)
                            std::cout << ", "; // Asegura que no haya coma después del último valor
                    }
                    std::cout << "]" << std::endl;

                    switch (ndim)
                    {
                    case 1:
                    {
                        // Imprime un vector 1D
                        std::cout << "[";
                        for (int i = 0; i < shape[0]; ++i)
                        {
                            std::cout << data[i];
                            if (i < shape[0] - 1)
                                std::cout << ", "; // No coma al final
                        }
                        std::cout << "]";
                        break;
                    }
                    case 2:
                    {
                        // Imprime una matriz 2D
                        std::cout << "[";
                        for (int i = 0; i < shape[0]; ++i)
                        {
                            std::cout << "[";
                            for (int j = 0; j < shape[1]; ++j)
                            {
                                std::cout << data[i * strds[0] + j * strds[1]];
                                if (j < shape[1] - 1)
                                    std::cout << ", "; // No coma al final de la fila
                            }
                            std::cout << "]";
                            if (i < shape[0] - 1)
                                std::cout << ", "; // No coma al final de la matriz
                        }
                        std::cout << "]";
                        break;
                    }
                    case 3:
                    {
                        // Imprime un tensor 3D
                        std::cout << "[";
                        for (int i = 0; i < shape[0]; ++i) // "Capas"
                        {
                            std::cout << "[";
                            for (int j = 0; j < shape[1]; ++j) // Filas
                            {
                                std::cout << "[";
                                for (int k = 0; k < shape[2]; ++k) // Columnas
                                {
                                    std::cout << data[i * strds[0] + j * strds[1] + k];
                                    if (k < shape[2] - 1)
                                        std::cout << ", "; // No coma al final de la columna
                                }
                                std::cout << "]";
                                if (j < shape[1] - 1)
                                    std::cout << ", "; // No coma al final de la fila
                            }
                            std::cout << "]";
                            if (i < shape[0] - 1)
                                std::cout << ", "; // No coma al final de la capa
                        }
                        std::cout << "]";
                        break;
                    }
                    case 4:
                    {
                        // Imprime un tensor 4D
                        std::cout << "[";
                        for (int i = 0; i < shape[0]; ++i) // "Bloques"
                        {
                            std::cout << "[";
                            for (int j = 0; j < shape[1]; ++j) // "Capas"
                            {
                                std::cout << "[";
                                for (int k = 0; k < shape[2]; ++k) // Filas
                                {
                                    std::cout << "[";
                                    for (int l = 0; l < shape[3]; ++l) // Columnas
                                    {
                                        std::cout << data[i * strds[0] + j * strds[1] + k * strds[2] + l];
                                        if (l < shape[3] - 1)
                                            std::cout << ", "; // No coma al final de la columna
                                    }
                                    std::cout << "]";
                                    if (k < shape[2] - 1)
                                        std::cout << ", "; // No coma al final de la fila
                                }
                                std::cout << "]";
                                if (j < shape[1] - 1)
                                    std::cout << ", "; // No coma al final de la capa
                            }
                            std::cout << "]";
                            if (i < shape[0] - 1)
                                std::cout << ", "; // No coma al final del bloque
                        }
                        std::cout << "]";
                        break;
                    }
                    default:
                        std::cerr << "Unsupported number of dimensions!" << std::endl;
                        break;
                    }
                    std::cout << std::endl;
                }

                float &operator()(int i)
                {
                    return data[i];
                }

                const float &operator()(int i) const
                {
                    return data[i];
                }

                float &operator()(int i, int j)
                {
                    return data[i * strds[0] + j * strds[1]];
                }

                const float &operator()(int i, int j) const
                {
                    return data[i * strds[0] + j * strds[1]];
                }

                float &operator()(int i, int j, int k)
                {
                    return data[i * strds[0] + j * strds[1] + k * strds[2]];
                }

                const float &operator()(int i, int j, int k) const
                {
                    return data[i * strds[0] + j * strds[1] + k * strds[2]];
                }

                float &operator()(int i, int j, int k, int l)
                {
                    return data[i * strds[0] + j * strds[1] + k * strds[2] + l * strds[3]];
                }

                const float &operator()(int i, int j, int k, int l) const
                {
                    return data[i * strds[0] + j * strds[1] + k * strds[2] + l * strds[3]];
                }

        */
    };

    template <>
    class Tensor<NN::GPU> : public TensorBase
    {
    public:
        GPUBuffer gpu_buffer;

        Tensor() : TensorBase() {}
        Tensor(int i) : TensorBase(i) {}
        Tensor(int i, int j) : TensorBase(i, j) {}
        Tensor(int i, int j, int k) : TensorBase(i, j, k) {}
        Tensor(int i, int j, int k, int l) : TensorBase(i, j, k, l) {}
        Tensor(int *shape, int ndim) : TensorBase(shape, ndim) {}
        Tensor(const Tensor &other);
    };

    template class Tensor<NN::CPU>;
    template class Tensor<NN::GPU>;
}