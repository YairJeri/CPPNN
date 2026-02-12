#pragma once

#include "TensorDevice.h"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <iostream>

namespace NN
{
    class MNISTLoader
    {
    private:
        static uint32_t swapEndian(uint32_t val)
        {
            return ((val >> 24) & 0x000000FF) |
                   ((val >> 8)  & 0x0000FF00) |
                   ((val << 8)  & 0x00FF0000) |
                   ((val << 24) & 0xFF000000);
        }

    public:
        // Carga imágenes en formato normalizado [0.0, 1.0] -> Tensor<NN::CPU> shape: [N, 784]
        static Tensor<NN::CPU> loadImages(const std::string &path, int max_samples = -1)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open())
                throw std::runtime_error("No se pudo abrir el archivo de imágenes: " + path);

            uint32_t magic_number = 0, num_images = 0, rows = 0, cols = 0;
            file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
            file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
            file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
            file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

            magic_number = swapEndian(magic_number);
            num_images = swapEndian(num_images);
            rows = swapEndian(rows);
            cols = swapEndian(cols);

            if (magic_number != 2051)
                throw std::runtime_error("Número mágico de imágenes MNIST inválido: " + std::to_string(magic_number));

            int count = (max_samples > 0 && max_samples < static_cast<int>(num_images)) ? max_samples : static_cast<int>(num_images);
            int image_size = rows * cols; // 784

            Tensor<NN::CPU> images(count, image_size);
            std::vector<uint8_t> buffer(image_size);

            for (int i = 0; i < count; ++i)
            {
                file.read(reinterpret_cast<char *>(buffer.data()), image_size);
                for (int j = 0; j < image_size; ++j)
                {
                    // Normalización de píxel [0, 255] -> [0.0, 1.0]
                    images(i, j) = static_cast<float>(buffer[j]) / 255.0f;
                }
            }

            return images;
        }

        // Carga etiquetas categóricas -> Tensor<NN::CPU> shape: [N] (con valores 0..9)
        static Tensor<NN::CPU> loadLabels(const std::string &path, int max_samples = -1)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open())
                throw std::runtime_error("No se pudo abrir el archivo de etiquetas: " + path);

            uint32_t magic_number = 0, num_labels = 0;
            file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
            file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

            magic_number = swapEndian(magic_number);
            num_labels = swapEndian(num_labels);

            if (magic_number != 2049)
                throw std::runtime_error("Número mágico de etiquetas MNIST inválido: " + std::to_string(magic_number));

            int count = (max_samples > 0 && max_samples < static_cast<int>(num_labels)) ? max_samples : static_cast<int>(num_labels);

            Tensor<NN::CPU> labels(count);
            std::vector<uint8_t> buffer(count);
            file.read(reinterpret_cast<char *>(buffer.data()), count);

            for (int i = 0; i < count; ++i)
            {
                labels(i) = static_cast<float>(buffer[i]);
            }

            return labels;
        }

        // Convierte un vector de etiquetas [N] a One-Hot encoding [N, 10]
        static Tensor<NN::CPU> toOneHot(const Tensor<NN::CPU> &labels, int num_classes = 10)
        {
            int n = labels.getDim(0);
            Tensor<NN::CPU> one_hot(n, num_classes);
            one_hot.fill(0.0f);
            for (int i = 0; i < n; ++i)
            {
                int label = static_cast<int>(labels(i));
                if (label >= 0 && label < num_classes)
                    one_hot(i, label) = 1.0f;
            }
            return one_hot;
        }

        // Dibuja en consola un carácter ASCII art del dígito para verificar visualmente
        static void printAsciiDigit(const Tensor<NN::CPU> &images, int sample_index)
        {
            const char *shades = " .:-=+*#%@";
            std::cout << "--- Imagen de muestra #" << sample_index << " (28x28) ---\n";
            for (int r = 0; r < 28; ++r)
            {
                for (int c = 0; c < 28; ++c)
                {
                    float val = images(sample_index, r * 28 + c);
                    int idx = static_cast<int>(val * 9.0f);
                    if (idx < 0) idx = 0;
                    if (idx > 9) idx = 9;
                    std::cout << shades[idx] << shades[idx];
                }
                std::cout << "\n";
            }
        }
    };
}

