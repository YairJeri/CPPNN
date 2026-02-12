#pragma once

#include "TensorDevice.h"
#include "MNISTLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <random>

namespace NN
{
    namespace fs = std::filesystem;

    enum class DatasetKind
    {
        TabularCSV,
        MNISTImages,
        Unknown
    };

    struct GenericDatasetInfo
    {
        std::string folder_name;
        std::string full_path;
        DatasetKind kind = DatasetKind::Unknown;
        std::string primary_file;
        int num_samples = 0;
        int num_features = 0;
        int num_classes = 0;
        int image_channels = 1;
        int image_height = 28;
        int image_width = 28;
        std::vector<std::string> feature_names;
        std::unordered_map<int, std::string> id_to_label;
    };

    class UniversalDatasetLoader
    {
    public:
        // Escanea recursivamente la carpeta 'datasets/' y detecta qué tipo de dataset es cada subcarpeta
        static std::vector<GenericDatasetInfo> scanDatasets(const std::string &base_dir = "datasets")
        {
            std::vector<GenericDatasetInfo> list;
            if (!fs::exists(base_dir) || !fs::is_directory(base_dir))
                return list;

            for (const auto &entry : fs::directory_iterator(base_dir))
            {
                if (entry.is_directory())
                {
                    GenericDatasetInfo info;
                    info.folder_name = entry.path().filename().string();
                    info.full_path = entry.path().string();

                    // Detectar si es MNIST
                    bool has_mnist_train_img = fs::exists(entry.path() / "train-images.idx3-ubyte");
                    bool has_mnist_train_lbl = fs::exists(entry.path() / "train-labels.idx1-ubyte");

                    if (has_mnist_train_img && has_mnist_train_lbl)
                    {
                        info.kind = DatasetKind::MNISTImages;
                        info.primary_file = "train-images.idx3-ubyte";
                        info.num_samples = 60000;
                        info.num_features = 784; // 28x28
                        info.image_channels = 1;
                        info.image_height = 28;
                        info.image_width = 28;
                        info.num_classes = 10;
                        for (int i = 0; i < 10; ++i)
                            info.id_to_label[i] = std::to_string(i);
                        list.push_back(info);
                        continue;
                    }

                    // Buscar si hay algún archivo .csv dentro
                    for (const auto &f : fs::directory_iterator(entry.path()))
                    {
                        if (f.is_regular_file() && f.path().extension() == ".csv")
                        {
                            info.kind = DatasetKind::TabularCSV;
                            info.primary_file = f.path().string();
                            inspectCSV(info);
                            list.push_back(info);
                            break;
                        }
                    }
                }
            }
            return list;
        }

        // Inspecciona el encabezado y dimensiones de un CSV genérico
        static void inspectCSV(GenericDatasetInfo &info)
        {
            std::ifstream file(info.primary_file);
            if (!file.is_open())
                return;

            std::string header_line;
            if (!std::getline(file, header_line))
                return;

            std::stringstream ss(header_line);
            std::string col;
            std::vector<std::string> columns;
            while (std::getline(ss, col, ','))
            {
                // Limpiar espacios en blanco o retornos de carro
                col.erase(std::remove(col.begin(), col.end(), '\r'), col.end());
                col.erase(std::remove(col.begin(), col.end(), '\n'), col.end());
                columns.push_back(col);
            }

            if (columns.size() >= 2)
            {
                // Asumimos por convención la última columna como la etiqueta objetivo
                info.feature_names = std::vector<std::string>(columns.begin(), columns.end() - 1);
                info.num_features = static_cast<int>(info.feature_names.size());
            }

            // Contar muestras y clases
            std::unordered_map<std::string, int> label_map;
            std::string line;
            int count = 0;
            while (std::getline(file, line))
            {
                if (line.empty()) continue;
                count++;
                size_t last_comma = line.rfind(',');
                if (last_comma != std::string::npos)
                {
                    std::string label = line.substr(last_comma + 1);
                    label.erase(std::remove(label.begin(), label.end(), '\r'), label.end());
                    label.erase(std::remove(label.begin(), label.end(), '\n'), label.end());
                    if (!label.empty() && label_map.find(label) == label_map.end())
                    {
                        int id = static_cast<int>(label_map.size());
                        label_map[label] = id;
                        info.id_to_label[id] = label;
                    }
                }
            }

            info.num_samples = count;
            info.num_classes = std::max(1, static_cast<int>(label_map.size()));
        }

        // Carga un CSV genérico en tensores listos para entrenamiento
        static bool loadCSV(const GenericDatasetInfo &info, 
                            Tensor<NN::CPU> &X_train, Tensor<NN::CPU> &y_train_onehot,
                            Tensor<NN::CPU> &X_test, Tensor<NN::CPU> &y_test_raw,
                            float test_split = 0.2f)
        {
            std::ifstream file(info.primary_file);
            if (!file.is_open())
                return false;

            std::string line;
            std::getline(file, line); // Saltar header

            std::vector<std::vector<float>> all_x;
            std::vector<int> all_y;
            std::unordered_map<std::string, int> str_to_id;
            for (const auto &[id, name] : info.id_to_label)
                str_to_id[name] = id;

            while (std::getline(file, line))
            {
                if (line.empty()) continue;
                std::stringstream ss(line);
                std::string cell;
                std::vector<float> row;
                for (int c = 0; c < info.num_features; ++c)
                {
                    if (std::getline(ss, cell, ','))
                    {
                        try {
                            row.push_back(std::stof(cell));
                        } catch (...) {
                            row.push_back(0.0f);
                        }
                    }
                }

                std::string lbl;
                if (std::getline(ss, lbl, ','))
                {
                    lbl.erase(std::remove(lbl.begin(), lbl.end(), '\r'), lbl.end());
                    lbl.erase(std::remove(lbl.begin(), lbl.end(), '\n'), lbl.end());
                    int id = (str_to_id.find(lbl) != str_to_id.end()) ? str_to_id[lbl] : 0;
                    all_y.push_back(id);
                    all_x.push_back(row);
                }
            }

            size_t total = all_x.size();
            if (total == 0) return false;

            // Split train/test
            std::vector<size_t> indices(total);
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 g(42);
            std::shuffle(indices.begin(), indices.end(), g);

            size_t test_count = static_cast<size_t>(test_split * total);
            size_t train_count = total - test_count;

            X_train = Tensor<NN::CPU>(train_count, info.num_features);
            y_train_onehot = Tensor<NN::CPU>(train_count, info.num_classes);
            y_train_onehot.fill(0.0f);

            for (size_t i = 0; i < train_count; ++i)
            {
                size_t idx = indices[i];
                for (int f = 0; f < info.num_features; ++f)
                    X_train(i, f) = all_x[idx][f];
                y_train_onehot(i, all_y[idx]) = 1.0f;
            }

            X_test = Tensor<NN::CPU>(test_count, info.num_features);
            y_test_raw = Tensor<NN::CPU>(test_count);
            for (size_t i = 0; i < test_count; ++i)
            {
                size_t idx = indices[train_count + i];
                for (int f = 0; f < info.num_features; ++f)
                    X_test(i, f) = all_x[idx][f];
                y_test_raw(i) = static_cast<float>(all_y[idx]);
            }

            return true;
        }
    };
}

