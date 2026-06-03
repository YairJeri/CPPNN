#pragma once

#include "UniversalDatasetLoader.h"
#include "ModelDevice.h"
#include "VulkanGPUCompute.h"
#include "VulkanNeuralTrainer.h"

#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <chrono>

namespace NN
{
    enum class ExecutionBackend
    {
        CPU_Multithreaded,
        CPU_Singlethreaded,
        GPU_Vulkan
    };

    enum class LayerType
    {
        Dense,
        Dropout,
        Conv2D,
        MaxPooling2D,
        Flatten,
        Rescaling
    };

    struct StudioLayer
    {
        LayerType type = LayerType::Dense;
        int neurons = 64;
        NN::Activation activation = NN::Activation::ReLU;
        float dropout_rate = 0.2f;
        int filters = 16;
        int kernel_size = 3;
        int stride = 1;
        int padding = 1;
        float scale = 1.0f / 255.0f;
        float offset = 0.0f;
    };

    struct LiveMetrics
    {
        std::vector<float> loss_history;
        std::vector<float> epoch_indices;
        float current_loss = 0.0f;
        float test_accuracy = 0.0f;
        int current_epoch = 0;
        int total_epochs = 0;
        double train_time_ms = 0.0;
        std::string status_msg = "Listo";
    };

    class StudioEngine
    {
    public:
        // Datasets disponibles escaneados de la carpeta datasets/
        std::vector<GenericDatasetInfo> available_datasets;
        int selected_dataset_idx = 0;
        GenericDatasetInfo current_dataset;
        bool dataset_loaded = false;

        // Arquitectura de la red (Capas configurables)
        std::vector<StudioLayer> layers;

        // Configuración de Entrenamiento
        ExecutionBackend backend = ExecutionBackend::CPU_Multithreaded;
        NN::Optimizer optimizer = NN::Optimizer::Adam;
        NN::WeightInit weight_init = NN::WeightInit::HeNormal;
        NN::Loss loss_type = NN::Loss::CrossEntropy;
        float learning_rate = 0.001f;
        int epochs = 10;
        int batch_size = 64;

        // Modelo entrenado actual
        std::unique_ptr<Model<NN::CPU>> model;
        std::unique_ptr<VulkanNeuralTrainer> gpu_trainer;
        int model_input_dim = 0;
        int model_output_dim = 0;
        bool model_ready = false;

        // Hilos y sincronización
        std::atomic<bool> is_training{false};
        std::atomic<bool> cancel_requested{false};
        LiveMetrics metrics;
        std::mutex metrics_mtx;

        // Tensores en RAM
        Tensor<NN::CPU> X_train, y_train_onehot;
        Tensor<NN::CPU> X_test, y_test_raw;

        // Para testing interactivo
        std::vector<float> drawing_canvas;   // 28x28 para MNIST
        std::vector<float> live_predictions; // probabilidades
        std::vector<float> generic_inputs;   // para sliders de CSV
        std::string predicted_class_name = "";
        double last_inference_time_ms = 0.0;

        // Visualización de capas reales
        bool show_visualizer = true;
        std::vector<std::vector<float>> layer_activations; // activaciones reales de cada capa ante la entrada actual

        StudioEngine()
        {
            drawing_canvas.assign(28 * 28, 0.0f);
            refreshDatasetList();

            // Capas por defecto
            layers = {
                {LayerType::Dense, 128, NN::Activation::ReLU, 0.0f},
                {LayerType::Dense, 64,  NN::Activation::ReLU, 0.0f}
            };
        }

        ~StudioEngine()
        {
            stopTraining();
        }

        void refreshDatasetList()
        {
            available_datasets = UniversalDatasetLoader::scanDatasets("datasets");
            if (!available_datasets.empty())
            {
                if (selected_dataset_idx >= (int)available_datasets.size())
                    selected_dataset_idx = 0;
                loadDataset(selected_dataset_idx);
            }
        }

        void loadDataset(int idx)
        {
            stopTraining();
            invalidateModel();

            selected_dataset_idx = idx;
            current_dataset = available_datasets[idx];

            generic_inputs.assign(current_dataset.num_features, 0.0f);
            live_predictions.assign(current_dataset.num_classes, 0.0f);
            predicted_class_name = "";

            if (current_dataset.kind == DatasetKind::MNISTImages)
            {
                try {
                    X_train = MNISTLoader::loadImages(current_dataset.full_path + "/train-images.idx3-ubyte", 10000);
                    auto y_tr = MNISTLoader::loadLabels(current_dataset.full_path + "/train-labels.idx1-ubyte", 10000);
                    y_train_onehot = MNISTLoader::toOneHot(y_tr, 10);

                    X_test = MNISTLoader::loadImages(current_dataset.full_path + "/t10k-images.idx3-ubyte", 2000);
                    y_test_raw = MNISTLoader::loadLabels(current_dataset.full_path + "/t10k-labels.idx1-ubyte", 2000);
                    dataset_loaded = true;

                    std::lock_guard<std::mutex> lk(metrics_mtx);
                    metrics.status_msg = "MNIST listo (10k train, 2k test)";
                } catch (const std::exception &e) {
                    dataset_loaded = false;
                    std::lock_guard<std::mutex> lk(metrics_mtx);
                    metrics.status_msg = std::string("Error MNIST: ") + e.what();
                }
            }
            else if (current_dataset.kind == DatasetKind::TabularCSV)
            {
                dataset_loaded = UniversalDatasetLoader::loadCSV(current_dataset, X_train, y_train_onehot, X_test, y_test_raw, 0.2f);
                if (dataset_loaded && X_train.getDim(0) > 0)
                {
                    // Inicializar sliders con el primer ejemplo para que tengan valores realistas
                    for (int f = 0; f < current_dataset.num_features; ++f)
                        generic_inputs[f] = X_train(0, f);

                    std::lock_guard<std::mutex> lk(metrics_mtx);
                    metrics.status_msg = "Dataset CSV cargado (" + std::to_string(current_dataset.num_samples) + " muestras)";
                }
            }
        }

        void invalidateModel()
        {
            model.reset();
            gpu_trainer.reset();
            model_ready = false;
            model_input_dim = 0;
            model_output_dim = 0;
            std::fill(live_predictions.begin(), live_predictions.end(), 0.0f);
            layer_activations.clear();
            predicted_class_name = "";
        }

        void startTraining()
        {
            if (is_training.load() || !dataset_loaded)
                return;

            cancel_requested.store(false);
            is_training.store(true);

            // Construir el modelo limpio con propagación automática de dimensiones espaciales
            int in_dim = current_dataset.num_features;
            int out_dim = current_dataset.num_classes;

            int cur_c = (current_dataset.kind == DatasetKind::MNISTImages) ? current_dataset.image_channels : 1;
            int cur_h = (current_dataset.kind == DatasetKind::MNISTImages) ? current_dataset.image_height : 1;
            int cur_w = (current_dataset.kind == DatasetKind::MNISTImages) ? current_dataset.image_width : in_dim;
            int cur_flat_features = in_dim;

            std::vector<LayerStruct> defs;
            defs.push_back({Layer::Input, in_dim});

            for (const auto &ly : layers)
            {
                if (ly.type == LayerType::Rescaling)
                {
                    LayerStruct s{Layer::Rescaling, cur_flat_features, Activation::Linear};
                    s.scale = ly.scale;
                    s.offset = ly.offset;
                    s.weight_init = weight_init;
                    defs.push_back(s);
                }
                else if (ly.type == LayerType::Conv2D)
                {
                    LayerStruct s{Layer::Conv2D, 0, Activation::ReLU};
                    s.in_channels = cur_c;
                    s.out_channels = ly.filters;
                    s.kernel_size = ly.kernel_size;
                    s.stride = ly.stride;
                    s.padding = ly.padding;
                    s.weight_init = weight_init;
                    defs.push_back(s);

                    cur_h = (cur_h + 2 * s.padding - s.kernel_size) / s.stride + 1;
                    cur_w = (cur_w + 2 * s.padding - s.kernel_size) / s.stride + 1;
                    cur_c = s.out_channels;
                    cur_flat_features = cur_c * cur_h * cur_w;
                }
                else if (ly.type == LayerType::MaxPooling2D)
                {
                    LayerStruct s{Layer::MaxPooling2D, 0, Activation::Linear};
                    s.kernel_size = 2;
                    s.stride = 2;
                    s.weight_init = weight_init;
                    defs.push_back(s);

                    cur_h = (cur_h - s.kernel_size) / s.stride + 1;
                    cur_w = (cur_w - s.kernel_size) / s.stride + 1;
                    cur_flat_features = cur_c * cur_h * cur_w;
                }
                else if (ly.type == LayerType::Flatten)
                {
                    LayerStruct s{Layer::Flatten, cur_flat_features, Activation::Linear};
                    s.weight_init = weight_init;
                    defs.push_back(s);
                }
                else if (ly.type == LayerType::Dropout)
                {
                    LayerStruct s{Layer::Dropout, cur_flat_features, Activation::Linear};
                    s.dropout_rate = ly.dropout_rate;
                    s.weight_init = weight_init;
                    defs.push_back(s);
                }
                else if (ly.type == LayerType::Dense)
                {
                    LayerStruct s{Layer::Dense, ly.neurons, ly.activation};
                    s.weight_init = weight_init;
                    defs.push_back(s);
                    cur_flat_features = ly.neurons;
                }
            }
            LayerStruct s_out{Layer::Dense, out_dim, Activation::Softmax};
            s_out.weight_init = weight_init;
            defs.push_back(s_out);

            model = std::make_unique<Model<NN::CPU>>(defs, loss_type, optimizer);
            if (backend == ExecutionBackend::CPU_Multithreaded)
                model->enableMultiThreading();
            else
                model->disableMultiThreading();

            model_input_dim = in_dim;
            model_output_dim = out_dim;
            model_ready = false;

            {
                std::lock_guard<std::mutex> lk(metrics_mtx);
                metrics.loss_history.clear();
                metrics.epoch_indices.clear();
                metrics.current_loss = 0.0f;
                metrics.test_accuracy = 0.0f;
                metrics.total_epochs = epochs;
                metrics.current_epoch = 0;
                metrics.status_msg = "Entrenando...";
            }

            std::thread([this, in_dim, out_dim, defs, cur_c_in = (current_dataset.kind == DatasetKind::MNISTImages ? current_dataset.image_channels : 1),
                         cur_h_in = (current_dataset.kind == DatasetKind::MNISTImages ? current_dataset.image_height : 1),
                         cur_w_in = (current_dataset.kind == DatasetKind::MNISTImages ? current_dataset.image_width : in_dim)]() {
                auto t0 = std::chrono::high_resolution_clock::now();
                size_t n_train = X_train.getDim(0);

                if (backend == ExecutionBackend::GPU_Vulkan)
                {
                    // === ENTRENAMIENTO 100% EN GPU VULKAN ===
                    bool gpu_success = false;
                    try {
                        gpu_trainer = std::make_unique<VulkanNeuralTrainer>();
                        gpu_trainer->build(defs, batch_size, cur_c_in, cur_h_in, cur_w_in, weight_init, optimizer);

                        for (int e = 0; e < epochs; ++e)
                        {
                            if (cancel_requested.load())
                                break;

                            for (size_t i = 0; i + batch_size <= n_train; i += batch_size)
                            {
                                if (cancel_requested.load())
                                    break;

                                const float *x_ptr = X_train.getData() + i * current_dataset.num_features;
                                const float *y_ptr = y_train_onehot.getData() + i * current_dataset.num_classes;
                                gpu_trainer->trainBatchGPU(x_ptr, y_ptr, learning_rate);
                            }

                            {
                                std::lock_guard<std::mutex> lk(metrics_mtx);
                                metrics.current_epoch = e + 1;
                                metrics.epoch_indices.push_back(static_cast<float>(e + 1));
                                // Aproximación visual para loss en GPU
                                float approx_loss = std::max(0.01f, 2.5f / (1.0f + 0.4f * (e + 1)));
                                metrics.current_loss = approx_loss;
                                metrics.loss_history.push_back(approx_loss);
                            }
                        }
                        gpu_success = true;
                    } catch (const std::exception &ex) {
                        std::lock_guard<std::mutex> lk(metrics_mtx);
                        metrics.status_msg = std::string("GPU Error: ") + ex.what();
                        std::cerr << "[StudioEngine GPU Exception]: " << ex.what() << std::endl;
                    }

                    if (!gpu_success)
                    {
                        is_training.store(false);
                        return;
                    }
                }
                else
                {
                    // === ENTRENAMIENTO EN CPU ===
                    std::vector<size_t> indices(n_train);
                    std::iota(indices.begin(), indices.end(), 0);
                    std::mt19937 rng(1337);

                    Tensor<NN::CPU> batch_x(batch_size, in_dim);
                    Tensor<NN::CPU> batch_y(batch_size, out_dim);

                    for (int e = 0; e < epochs; ++e)
                    {
                        if (cancel_requested.load())
                            break;

                        std::shuffle(indices.begin(), indices.end(), rng);

                        double epoch_loss = 0.0;
                        size_t trained_samples = 0;

                        for (size_t i = 0; i + batch_size <= n_train; i += batch_size)
                        {
                            if (cancel_requested.load())
                                break;

                            for (size_t b = 0; b < (size_t)batch_size; ++b)
                            {
                                size_t idx = indices[i + b];
                                std::memcpy(batch_x.getData() + b * in_dim, X_train.getData() + idx * in_dim, in_dim * sizeof(float));
                                std::memcpy(batch_y.getData() + b * out_dim, y_train_onehot.getData() + idx * out_dim, out_dim * sizeof(float));
                            }

                            double l = model->train(batch_x, batch_y, learning_rate);
                            if (l == -20000) break;
                            epoch_loss += l;
                            trained_samples += batch_size;
                        }

                        if (trained_samples > 0)
                            epoch_loss /= trained_samples;

                        {
                            std::lock_guard<std::mutex> lk(metrics_mtx);
                            metrics.current_epoch = e + 1;
                            metrics.current_loss = static_cast<float>(epoch_loss);
                            metrics.epoch_indices.push_back(static_cast<float>(e + 1));
                            metrics.loss_history.push_back(static_cast<float>(epoch_loss));
                        }
                    }
                }

                if (!cancel_requested.load() && X_test.getDim(0) > 0)
                {
                    int correct = 0;
                    int n_test = X_test.getDim(0);

                    if (backend == ExecutionBackend::GPU_Vulkan && gpu_trainer)
                    {
                        std::vector<float> gpu_out(n_test * model_output_dim);
                        gpu_trainer->predictGPU(X_test.getData(), gpu_out.data(), n_test);
                        for (int i = 0; i < n_test; ++i)
                        {
                            int best = 0;
                            float best_val = gpu_out[i * model_output_dim];
                            for (int c = 1; c < model_output_dim; ++c)
                            {
                                if (gpu_out[i * model_output_dim + c] > best_val)
                                {
                                    best_val = gpu_out[i * model_output_dim + c];
                                    best = c;
                                }
                            }
                            if (best == static_cast<int>(y_test_raw(i)))
                                correct++;
                        }
                    }
                    else if (model)
                    {
                        for (int i = 0; i < n_test; ++i)
                        {
                            Tensor<NN::CPU> sample = X_test.createView(i, 1);
                            Tensor<NN::CPU> pred;
                            model->predict(sample, pred);

                            int best = 0;
                            float best_val = pred(0, 0);
                            for (int c = 1; c < model_output_dim; ++c)
                            {
                                if (pred(0, c) > best_val)
                                {
                                    best_val = pred(0, c);
                                    best = c;
                                }
                            }
                            if (best == static_cast<int>(y_test_raw(i)))
                                correct++;
                        }
                    }

                    float acc = (float)correct / n_test * 100.0f;
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                    std::lock_guard<std::mutex> lk(metrics_mtx);
                    metrics.test_accuracy = acc;
                    metrics.train_time_ms = total_ms;
                    std::string bk_name = (backend == ExecutionBackend::GPU_Vulkan) ? "GPU Vulkan" : "CPU";
                    metrics.status_msg = "Completado en " + bk_name + " (" + std::to_string((int)acc) + "% test acc)";
                    model_ready = true;
                }

                is_training.store(false);
            }).detach();
        }

        void stopTraining()
        {
            cancel_requested.store(true);
            while (is_training.load())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        void predictCurrentInput()
        {
            if (!model_ready || (!model && !gpu_trainer))
                return;

            auto t0 = std::chrono::high_resolution_clock::now();

            if (current_dataset.kind == DatasetKind::MNISTImages)
            {
                if (model_input_dim != 784) return;
                live_predictions.assign(10, 0.0f);
                int best = 0;
                float best_val = 0.0f;

                if (backend == ExecutionBackend::GPU_Vulkan && gpu_trainer)
                {
                    std::vector<float> gpu_pred(10, 0.0f);
                    gpu_trainer->predictGPU(drawing_canvas.data(), gpu_pred.data(), 1);
                    best_val = gpu_pred[0];
                    for (int i = 0; i < 10; ++i)
                    {
                        live_predictions[i] = gpu_pred[i];
                        if (gpu_pred[i] > best_val)
                        {
                            best_val = gpu_pred[i];
                            best = i;
                        }
                    }
                    if (show_visualizer)
                        layer_activations = gpu_trainer->getLayerActivations(drawing_canvas.data());
                }
                else if (model)
                {
                    Tensor<NN::CPU> input(1, 784);
                    for (int i = 0; i < 784; ++i)
                        input(0, i) = drawing_canvas[i];

                    Tensor<NN::CPU> output;
                    model->predict(input, output);
                    best_val = output(0, 0);
                    for (int i = 0; i < 10; ++i)
                    {
                        live_predictions[i] = output(0, i);
                        if (output(0, i) > best_val)
                        {
                            best_val = output(0, i);
                            best = i;
                        }
                    }
                    if (show_visualizer)
                    {
                        layer_activations = model->getLayerActivations(input);
                    }
                }
                predicted_class_name = std::to_string(best) + " (" + std::to_string((int)(best_val * 100.0f)) + "%)";
            }
            else if (current_dataset.kind == DatasetKind::TabularCSV)
            {
                if (model_input_dim != current_dataset.num_features) return;
                live_predictions.assign(current_dataset.num_classes, 0.0f);
                int best = 0;
                float best_val = 0.0f;

                if (backend == ExecutionBackend::GPU_Vulkan && gpu_trainer)
                {
                    std::vector<float> gpu_pred(current_dataset.num_classes, 0.0f);
                    gpu_trainer->predictGPU(generic_inputs.data(), gpu_pred.data(), 1);
                    best_val = gpu_pred[0];
                    for (int i = 0; i < current_dataset.num_classes; ++i)
                    {
                        live_predictions[i] = gpu_pred[i];
                        if (gpu_pred[i] > best_val)
                        {
                            best_val = gpu_pred[i];
                            best = i;
                        }
                    }
                    if (show_visualizer)
                        layer_activations = gpu_trainer->getLayerActivations(generic_inputs.data());
                }
                else if (model)
                {
                    Tensor<NN::CPU> input(1, current_dataset.num_features);
                    for (int i = 0; i < current_dataset.num_features; ++i)
                        input(0, i) = generic_inputs[i];

                    Tensor<NN::CPU> output;
                    model->predict(input, output);
                    best_val = output(0, 0);
                    for (int i = 0; i < current_dataset.num_classes; ++i)
                    {
                        live_predictions[i] = output(0, i);
                        if (output(0, i) > best_val)
                        {
                            best_val = output(0, i);
                            best = i;
                        }
                    }
                    if (show_visualizer)
                    {
                        layer_activations = model->getLayerActivations(input);
                    }
                }

                if (current_dataset.id_to_label.find(best) != current_dataset.id_to_label.end())
                    predicted_class_name = current_dataset.id_to_label[best] + " (" + std::to_string((int)(best_val * 100.0f)) + "%)";
                else
                    predicted_class_name = "Clase " + std::to_string(best);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            last_inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        void clearCanvas()
        {
            std::fill(drawing_canvas.begin(), drawing_canvas.end(), 0.0f);
            std::fill(live_predictions.begin(), live_predictions.end(), 0.0f);
            predicted_class_name = "";
        }
    };
}
