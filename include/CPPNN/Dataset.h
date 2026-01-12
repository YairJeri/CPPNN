#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include "TensorDevice.h"

namespace NN
{

    class Dataset
    {
    public:
        std::vector<std::vector<float>> data;
        std::vector<std::string> raw_labels; // etiquetas categóricas originales
        std::unordered_map<std::string, int> label_to_id;
        std::unordered_map<int, std::string> id_to_label;

        int label_column = -1;

        Dataset() = default;

        Dataset(const std::string &filename, int label_col = -1)
        {
            load_csv(filename, label_col);
        }

        void load_csv(const std::string &filename, int label_col = -1, bool has_header = true)
        {
            std::ifstream file(filename);
            if (!file)
                throw std::runtime_error("No se pudo abrir el archivo: " + filename);

            std::string line;

            // Saltar la primera línea si tiene encabezado
            if (has_header && std::getline(file, line))
            {
                // no hacer nada, solo descartar
            }

            while (std::getline(file, line))
            {
                std::stringstream ss(line);
                std::string cell;
                std::vector<float> row;
                int col_index = 0;

                while (std::getline(ss, cell, ','))
                {
                    if (col_index == label_col)
                    {
                        raw_labels.push_back(cell);
                    }
                    else
                    {
                        try
                        {
                            row.push_back(std::stof(cell));
                        }
                        catch (...)
                        {
                            throw std::runtime_error("Error al convertir a float: '" + cell + "'");
                        }
                    }
                    col_index++;
                }

                data.push_back(row);
            }

            label_column = label_col;
            encode_labels();
        }

        void encode_labels()
        {
            for (const auto &label : raw_labels)
            {
                if (label_to_id.find(label) == label_to_id.end())
                {
                    int id = static_cast<int>(label_to_id.size());
                    label_to_id[label] = id;
                    id_to_label[id] = label;
                }
            }
        }

        Tensor<NN::CPU> features_tensor() const
        {
            int n = data.size();
            int d = data[0].size();
            Tensor<NN::CPU> X(n, d);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < d; ++j)
                    X(i, j) = data[i][j];
            return X;
        }

        Tensor<NN::CPU> labels_tensor() const
        {
            int n = raw_labels.size();
            Tensor<NN::CPU> y(n);
            for (int i = 0; i < n; ++i)
                y(i) = static_cast<float>(label_to_id.at(raw_labels[i]));
            return y;
        }

        std::pair<Dataset, Dataset> split(float test_ratio = 0.2) const
        {
            std::vector<size_t> indices(data.size());
            std::iota(indices.begin(), indices.end(), 0); // Llenar con 0, 1, ..., n-1

            // Mezclar aleatoriamente
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            size_t test_size = static_cast<size_t>(test_ratio * data.size());

            Dataset train_ds, test_ds;
            train_ds.label_column = test_ds.label_column = label_column;
            train_ds.label_to_id = test_ds.label_to_id = label_to_id;
            train_ds.id_to_label = test_ds.id_to_label = id_to_label;

            for (size_t i = 0; i < indices.size(); ++i)
            {
                size_t idx = indices[i];
                if (i < test_size)
                {
                    test_ds.data.push_back(data[idx]);
                    test_ds.raw_labels.push_back(raw_labels[idx]);
                }
                else
                {
                    train_ds.data.push_back(data[idx]);
                    train_ds.raw_labels.push_back(raw_labels[idx]);
                }
            }

            return {train_ds, test_ds};
        }

        void print_info() const
        {
            std::cout << "Samples: " << data.size() << ", Features: " << (data.empty() ? 0 : data[0].size()) << "\n";
            std::cout << "Classes:\n";
            for (const auto &pair : label_to_id)
                std::cout << "  " << pair.first << " -> " << pair.second << "\n";
        }
    };

}