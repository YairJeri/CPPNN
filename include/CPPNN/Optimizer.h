#pragma once
#include "TensorDevice.h"
#include <unordered_map>

namespace NN
{
    class OptimizerBase
    {
    public:
        virtual void update(Tensor<NN::CPU> &weights, Tensor<NN::CPU> &grads, float lr) = 0;
    };

    class SGD : public OptimizerBase
    {
    public:
        void update(Tensor<NN::CPU> &weights, Tensor<NN::CPU> &grads, float lr) override
        {
            for (int i = 0; i < weights.getSize(); ++i)
                weights(i) -= grads(i) * lr;
        }
    };

    class SGDMomentum : public OptimizerBase
    {
    public:
        float momentum;
        std::unordered_map<Tensor<NN::CPU> *, Tensor<NN::CPU>> velocities;

        SGDMomentum(float momentum = 0.9f)
            : momentum(momentum) {}

        void update(Tensor<NN::CPU> &weights, Tensor<NN::CPU> &grads, float lr) override
        {
            if (velocities.find(&weights) == velocities.end())
            {
                velocities[&weights] = Tensor<NN::CPU>(weights);
                velocities[&weights].fill(0.0f);
            }

            Tensor<NN::CPU> &velocity = velocities[&weights];
            float *w_ptr = weights.getData();
            const float *g_ptr = grads.getData();
            float *v_ptr = velocity.getData();
            int sz = weights.getSize();

            for (int i = 0; i < sz; ++i)
            {
                v_ptr[i] = momentum * v_ptr[i] - lr * g_ptr[i];
                w_ptr[i] += v_ptr[i];
            }
        }
    };

    class Adam : public OptimizerBase
    {
    public:
        float beta1;
        float beta2;
        float epsilon;
        std::unordered_map<Tensor<NN::CPU> *, int> t_steps;
        std::unordered_map<Tensor<NN::CPU> *, Tensor<NN::CPU>> m;
        std::unordered_map<Tensor<NN::CPU> *, Tensor<NN::CPU>> v;

        Adam(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
            : beta1(beta1), beta2(beta2), epsilon(epsilon) {}

        void update(Tensor<NN::CPU> &weights, Tensor<NN::CPU> &grads, float lr) override
        {
            if (m.find(&weights) == m.end())
            {
                m[&weights] = Tensor<NN::CPU>(weights);
                v[&weights] = Tensor<NN::CPU>(weights);
                m[&weights].fill(0.0f);
                v[&weights].fill(0.0f);
                t_steps[&weights] = 0;
            }

            Tensor<NN::CPU> &m_ = m[&weights];
            Tensor<NN::CPU> &v_ = v[&weights];
            int t = ++t_steps[&weights];

            float *w_ptr = weights.getData();
            const float *g_ptr = grads.getData();
            float *m_ptr = m_.getData();
            float *v_ptr = v_.getData();
            int sz = weights.getSize();

            float b1_pow = std::pow(beta1, t);
            float b2_pow = std::pow(beta2, t);
            float lr_t = lr * (std::sqrt(1.0f - b2_pow) / (1.0f - b1_pow));

            for (int i = 0; i < sz; ++i)
            {
                float g = g_ptr[i];
                m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
                v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
                w_ptr[i] -= lr_t * m_ptr[i] / (std::sqrt(v_ptr[i]) + epsilon);
            }
        }
    };
}
