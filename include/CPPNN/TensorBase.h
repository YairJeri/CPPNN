#pragma once

#include <iostream>
#include <cmath>

namespace NN
{
    class TensorBase
    {
    protected:
        int shape[4] = {1, 1, 1, 1};
        int strds[4] = {1, 1, 1, 1};
        int ndim = 0;
        bool own_data = true;
        float *data = nullptr;
        int size = 0;

    public:
        TensorBase();
        TensorBase(int i);
        TensorBase(int i, int j);
        TensorBase(int i, int j, int k);
        TensorBase(int i, int j, int k, int l);
        TensorBase(int *shape, int ndim);
        virtual ~TensorBase();

        void fill(float value);
        void generateWeights();

        size_t getSize() const;
        int getDim(int dim) const;
        int getNdim() const;
        int *getShape() const;
        float *getData();
    };
}