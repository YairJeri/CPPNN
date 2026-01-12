#include "CPPNN/TensorDevice.h"
#include <Eigen/Dense>

namespace NN
{
    Tensor<NN::CPU>::Tensor(const Tensor &other) : TensorBase()
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

    Tensor<NN::CPU> &Tensor<NN::CPU>::operator=(const Tensor &other)
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

    Tensor<NN::CPU> &Tensor<NN::CPU>::operator=(Tensor &&other) noexcept
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

    void Tensor<NN::CPU>::apply(float (*func)(float))
    {
        for (int i = 0; i < size; ++i)
            data[i] = func(data[i]);
    }

    Tensor<NN::CPU> Tensor<NN::CPU>::createView(size_t start, size_t count) const
    {
        if (start < 0 || count < 1 || start + count > shape[0])
            throw std::out_of_range("View out of range");

        Tensor view;
        view.ndim = ndim;
        view.size = (size / shape[0]) * count;
        view.shape[0] = count;
        for (int d = 1; d < ndim; ++d)
            view.shape[d] = shape[d];
        for (int d = 0; d < ndim; ++d)
            view.strds[d] = strds[d];

        size_t offset = start * strds[0];
        view.data = data + offset;
        view.own_data = false;

        return view;
    }

    Tensor<NN::CPU> Tensor<NN::CPU>::transposeView(int axis1, int axis2) const
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

    Tensor<NN::CPU> Tensor<NN::CPU>::transpose() const
    {
        Tensor result(shape[1], shape[0]);
        using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        Eigen::Map<const RowMajorMatrix> A(data, shape[0], shape[1]);
        Eigen::Map<RowMajorMatrix> B(result.data, shape[1], shape[0]);

        B.noalias() = A.transpose();
        return result;
    }

    Tensor<NN::CPU> Tensor<NN::CPU>::matmul(const Tensor &other) const
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

    Tensor<NN::CPU> Tensor<NN::CPU>::matmulThreaded(const Tensor &other) const
    {
        if (ndim != 2 || other.ndim != 2)
            throw std::invalid_argument("Tensors must be 2D for matmul");
        if (shape[1] != other.shape[0])
            throw std::invalid_argument("matmul: shape mismatch");

        Tensor result(shape[0], other.shape[1]);

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(this->data, shape[0], shape[1]);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(other.data, other.shape[0], other.shape[1]);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C(result.data, shape[0], other.shape[1]);

        C.noalias() = A * B;

        return result;
    }

    Tensor<NN::CPU> Tensor<NN::CPU>::forward(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &weights, const Tensor<NN::CPU> &biases)
    {
        Tensor result(input.shape[0], weights.shape[1]);

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(input.data, input.shape[0], input.shape[1]);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(weights.data, weights.shape[0], weights.shape[1]);

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C(result.data, result.shape[0], result.shape[1]);
        Eigen::Map<const Eigen::RowVectorXf> biasVec(biases.data, biases.shape[1]);

        C.noalias() = A * B;
        C.rowwise() += biasVec;

        return result;
    }

    Tensor<NN::CPU> &Tensor<NN::CPU>::broadcastAdd(const Tensor &other)
    {
        for (int i = 0; i < shape[0]; ++i)
            for (int j = 0; j < shape[1]; ++j)
                data[i * strds[0] + j * strds[1]] += other(j);
        return *this;
    }

    Tensor<NN::CPU> Tensor<NN::CPU>::operator*(float scalar) const
    {
        Tensor result(*this);
        for (int i = 0; i < size; ++i)
            result.data[i] = data[i] * scalar;
        return result;
    }

    Tensor<NN::CPU> &Tensor<NN::CPU>::operator*=(float scalar)
    {
        for (int i = 0; i < size; ++i)
            data[i] *= scalar;
        return *this;
    }

    Tensor<NN::CPU> Tensor<NN::CPU>::operator*(const Tensor &other) const
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

    Tensor<NN::CPU> &Tensor<NN::CPU>::operator*=(const Tensor &other)
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

    Tensor<NN::CPU> Tensor<NN::CPU>::operator+(const Tensor &other) const
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

    Tensor<NN::CPU> &Tensor<NN::CPU>::operator+=(const Tensor &other)
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

    Tensor<NN::CPU> Tensor<NN::CPU>::operator-(const Tensor &other) const
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

    Tensor<NN::CPU> &Tensor<NN::CPU>::operator-=(const Tensor &other)
    {
        if (shape[0] != other.shape[0] || shape[1] != other.shape[1] || shape[2] != other.shape[2] || shape[3] != other.shape[3])
            throw std::invalid_argument("-=: Tensors must have the same shape for element-wise subtraction");

        for (int i = 0; i < size; ++i)
        {
            data[i] -= other.data[i];
        }
        return *this;
    }

    void Tensor<NN::CPU>::print()
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

    float &Tensor<NN::CPU>::operator()(int i)
    {
        return data[i];
    }

    const float &Tensor<NN::CPU>::operator()(int i) const
    {
        return data[i];
    }

    float &Tensor<NN::CPU>::operator()(int i, int j)
    {
        return data[i * strds[0] + j * strds[1]];
    }

    const float &Tensor<NN::CPU>::operator()(int i, int j) const
    {
        return data[i * strds[0] + j * strds[1]];
    }

    float &Tensor<NN::CPU>::operator()(int i, int j, int k)
    {
        return data[i * strds[0] + j * strds[1] + k * strds[2]];
    }

    const float &Tensor<NN::CPU>::operator()(int i, int j, int k) const
    {
        return data[i * strds[0] + j * strds[1] + k * strds[2]];
    }

    float &Tensor<NN::CPU>::operator()(int i, int j, int k, int l)
    {
        return data[i * strds[0] + j * strds[1] + k * strds[2] + l * strds[3]];
    }

    const float &Tensor<NN::CPU>::operator()(int i, int j, int k, int l) const
    {
        return data[i * strds[0] + j * strds[1] + k * strds[2] + l * strds[3]];
    }
}