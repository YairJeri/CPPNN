#include "CPPNN/TensorDevice.h"

namespace NN
{

    TensorBase::TensorBase() {}

    TensorBase::TensorBase(int i) : ndim(1), size(i)
    {
        shape[0] = i;
        strds[0] = 1;
        data = new float[size];
        own_data = true;
    }

    TensorBase::TensorBase(int i, int j) : ndim(2), size(i * j)
    {
        shape[0] = i;
        shape[1] = j;
        strds[1] = 1;
        strds[0] = j;
        data = new float[size];
        own_data = true;
    }

    TensorBase::TensorBase(int i, int j, int k) : ndim(3), size(i * j * k)
    {
        shape[0] = i;
        shape[1] = j;
        shape[2] = k;
        strds[2] = 1;
        strds[1] = k;
        strds[0] = j * k;
        data = new float[size];
        own_data = true;
    }

    TensorBase::TensorBase(int i, int j, int k, int l) : ndim(4), size(i * j * k * l)
    {
        shape[0] = i;
        shape[1] = j;
        shape[2] = k;
        shape[3] = l;
        strds[3] = 1;
        strds[2] = l;
        strds[1] = k * l;
        strds[0] = j * k * l;
        data = new float[size];
        own_data = true;
    }

    TensorBase::TensorBase(int *shape, int ndim)
    {
        this->ndim = ndim;
        this->size = 1;
        for (int i = 0; i < ndim; ++i)
        {
            this->shape[i] = shape[i];
            this->strds[i] = 1;
            this->size *= shape[i];
        }
        data = new float[size];
        own_data = true;
    }

    TensorBase::~TensorBase()
    {
        if (own_data && data)
            delete[] data;
    }

    void TensorBase::fill(float value)
    {
        std::fill(data, data + size, value);
    }

    void TensorBase::generateWeights()
    {
        double limit = std::sqrt(6.0 / (shape[0] + shape[1]));
        for (int i = 0; i < shape[0]; ++i)
            for (int j = 0; j < shape[1]; ++j)
                data[i * strds[0] + j * strds[1]] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
    }

    size_t TensorBase::getSize() const { return size; }
    int TensorBase::getDim(int dim) const { return shape[dim]; }
    int TensorBase::getNdim() const { return ndim; }
    int *TensorBase::getShape() const { return (int *)shape; }
    float *TensorBase::getData() { return data; }
}
