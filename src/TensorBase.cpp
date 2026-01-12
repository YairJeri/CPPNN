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
        for (int i = 0; i < 4; ++i)
        {
            this->shape[i] = (i < ndim) ? shape[i] : 1;
            this->strds[i] = 1;
            if (i < ndim) this->size *= shape[i];
        }

        // Calcular strides row-major
        if (ndim == 1) {
            strds[0] = 1;
        } else if (ndim == 2) {
            strds[1] = 1;
            strds[0] = this->shape[1];
        } else if (ndim == 3) {
            strds[2] = 1;
            strds[1] = this->shape[2];
            strds[0] = this->shape[1] * this->shape[2];
        } else if (ndim == 4) {
            strds[3] = 1;
            strds[2] = this->shape[3];
            strds[1] = this->shape[2] * this->shape[3];
            strds[0] = this->shape[1] * this->shape[2] * this->shape[3];
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

    void TensorBase::generateWeights(WeightInit init, int custom_fan_in, int custom_fan_out)
    {
        if (!data || size <= 0) return;

        int fan_in = 1;
        int fan_out = 1;

        if (custom_fan_in > 0 && custom_fan_out > 0)
        {
            fan_in = custom_fan_in;
            fan_out = custom_fan_out;
        }
        else if (ndim == 4)
        {
            // Conv2D weights: [out_channels, in_channels, k_h, k_w]
            int k_area = shape[2] * shape[3];
            fan_in = shape[1] * k_area;
            fan_out = shape[0] * k_area;
        }
        else if (ndim == 2)
        {
            // Dense weights: [in_dim, out_dim]
            fan_in = shape[0];
            fan_out = shape[1];
        }
        else
        {
            fan_in = (shape[0] > 0) ? shape[0] : 1;
            fan_out = (shape[1] > 0) ? shape[1] : 1;
        }

        // Helper para distribución normal estándar usando Box-Muller
        auto rand_normal = []() -> float {
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979323846f * u2);
        };

        // Helper para uniforme [-limit, limit]
        auto rand_uniform = [](float limit) -> float {
            return (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * limit;
        };

        switch (init)
        {
        case WeightInit::HeNormal:
        {
            float stddev = std::sqrt(2.0f / (float)fan_in);
            for (int i = 0; i < size; ++i)
                data[i] = rand_normal() * stddev;
            break;
        }
        case WeightInit::HeUniform:
        {
            float limit = std::sqrt(6.0f / (float)fan_in);
            for (int i = 0; i < size; ++i)
                data[i] = rand_uniform(limit);
            break;
        }
        case WeightInit::GlorotUniform:
        {
            float limit = std::sqrt(6.0f / (float)(fan_in + fan_out));
            for (int i = 0; i < size; ++i)
                data[i] = rand_uniform(limit);
            break;
        }
        case WeightInit::GlorotNormal:
        {
            float stddev = std::sqrt(2.0f / (float)(fan_in + fan_out));
            for (int i = 0; i < size; ++i)
                data[i] = rand_normal() * stddev;
            break;
        }
        case WeightInit::LeCunNormal:
        {
            float stddev = std::sqrt(1.0f / (float)fan_in);
            for (int i = 0; i < size; ++i)
                data[i] = rand_normal() * stddev;
            break;
        }
        case WeightInit::RandomUniform:
        default:
        {
            float limit = 0.05f;
            for (int i = 0; i < size; ++i)
                data[i] = rand_uniform(limit);
            break;
        }
        }
    }

    size_t TensorBase::getSize() const { return size; }
    int TensorBase::getDim(int dim) const { return shape[dim]; }
    int TensorBase::getNdim() const { return ndim; }
    int *TensorBase::getShape() const { return (int *)shape; }
    float *TensorBase::getData() { return data; }
    const float *TensorBase::getData() const { return data; }
}
