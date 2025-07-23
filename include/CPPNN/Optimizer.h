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
                velocities[&weights] = Tensor<NN::CPU>(weights.getShape(), weights.getNdim());
                velocities[&weights].fill(0.0f);
            }

            Tensor<NN::CPU> &velocity = velocities[&weights];

            velocity = velocity * momentum - grads * lr;
            weights += velocity;
        }
    };

    class Adam : public OptimizerBase
    {
    public:
        float beta1;
        float beta2;
        float epsilon;
        int t;
        std::unordered_map<Tensor<NN::CPU> *, Tensor<NN::CPU>> m;
        std::unordered_map<Tensor<NN::CPU> *, Tensor<NN::CPU>> v;

        Adam(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
            : beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

        void update(Tensor<NN::CPU> &weights, Tensor<NN::CPU> &grads, float lr) override
        {
            if (m.find(&weights) == m.end())
            {
                m[&weights] = Tensor<NN::CPU>(weights.getShape(), weights.getNdim());
                v[&weights] = Tensor<NN::CPU>(weights.getShape(), weights.getNdim());
                m[&weights].fill(0.0f);
                v[&weights].fill(0.0f);
            }

            Tensor<NN::CPU> &m_ = m[&weights];
            Tensor<NN::CPU> &v_ = v[&weights];

            t++;

            for (int i = 0; i < weights.getSize(); ++i)
            {
                m_(i) = beta1 * m_(i) + (1.0f - beta1) * grads(i);
                v_(i) = beta2 * v_(i) + (1.0f - beta2) * grads(i) * grads(i);
            }

            float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));

            for (int i = 0; i < weights.getSize(); ++i)
            {
                weights(i) -= lr_t * m_(i) / (std::sqrt(v_(i)) + epsilon);
            }
        }
    };
}
