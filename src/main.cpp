#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <GLFW/glfw3.h>
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "implot.h"

#include "CPPNN/StudioEngine.h"

static void glfw_error_callback(int error, const char *description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// Renderiza la inspección de activaciones reales de cada capa ante la entrada actual
void RenderNetworkActivations(const NN::StudioEngine &engine, ImVec2 size)
{
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(14, 16, 22, 255), 6.0f);
    dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(45, 52, 70, 255), 6.0f);

    std::vector<int> layer_sizes;
    layer_sizes.push_back(engine.current_dataset.num_features);
    for (const auto &l : engine.layers)
    {
        if (l.type == NN::LayerType::Dense)
            layer_sizes.push_back(l.neurons);
    }
    layer_sizes.push_back(engine.current_dataset.num_classes);

    int num_layers = (int)layer_sizes.size();
    if (num_layers < 2) return;

    float layer_col_w = (size.x - 20.0f) / num_layers;

    for (int l = 0; l < num_layers; ++l)
    {
        float lx = pos.x + 10.0f + l * layer_col_w;
        float ly = pos.y + 26.0f;
        int n_neurons = layer_sizes[l];

        // Título de la capa
        char tag[64];
        if (l == 0) snprintf(tag, sizeof(tag), "Entrada (%d)", n_neurons);
        else if (l == num_layers - 1) snprintf(tag, sizeof(tag), "Salida (%d)", n_neurons);
        else snprintf(tag, sizeof(tag), "Capa %d (%d)", l, n_neurons);

        dl->AddText(ImVec2(lx + 4.0f, pos.y + 6.0f), IM_COL32(200, 220, 255, 230), tag);

        // Área de celdas para esta capa
        float avail_w = layer_col_w - 12.0f;
        float avail_h = size.y - 36.0f;

        // Calcular rejilla cuadrada/rectangular para representar las neuronas reales
        int cols = (int)std::ceil(std::sqrt((double)n_neurons));
        if (cols < 1) cols = 1;
        int rows = (n_neurons + cols - 1) / cols;

        float cell_w = std::max(2.0f, std::min(14.0f, (avail_w - (cols - 1)) / cols));
        float cell_h = std::max(2.0f, std::min(14.0f, (avail_h - (rows - 1)) / rows));
        float cell_sz = std::min(cell_w, cell_h);

        // Si tenemos activaciones calculadas para esta capa
        bool has_act = (l < (int)engine.layer_activations.size() && !engine.layer_activations[l].empty());

        for (int i = 0; i < n_neurons; ++i)
        {
            int r = i / cols;
            int c = i % cols;
            float px0 = lx + c * (cell_sz + 1.0f);
            float py0 = ly + r * (cell_sz + 1.0f);

            if (py0 + cell_sz > pos.y + size.y - 4.0f)
                break; // Fuera del área visible

            float act_val = 0.0f;
            if (has_act && i < (int)engine.layer_activations[l].size())
            {
                act_val = engine.layer_activations[l][i];
            }

            // Normalización de color: apagado (azul oscuro/gris) -> encendido (verde/amarillo brillante)
            float norm = std::min(1.0f, std::max(0.0f, act_val));
            ImU32 cell_col;
            if (norm > 0.01f)
            {
                int red = (int)(norm * 255);
                int green = (int)(255 - norm * 50);
                int blue = (int)(50 * (1.0f - norm));
                cell_col = IM_COL32(red, green, blue, 240);
            }
            else
            {
                cell_col = IM_COL32(28, 32, 42, 200);
            }

            dl->AddRectFilled(ImVec2(px0, py0), ImVec2(px0 + cell_sz, py0 + cell_sz), cell_col, 1.0f);
        }
    }

    ImGui::Dummy(size);
}

int main(int, char **)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow *window = glfwCreateWindow(1366, 850, "CPPNN Studio - Interactive Deep Learning Suite", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Tema Dark Studio profesional refinado
    ImGui::StyleColorsDark();
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 8.0f;
    style.ChildRounding = 8.0f;
    style.FrameRounding = 6.0f;
    style.PopupRounding = 6.0f;
    style.GrabRounding = 5.0f;
    style.ScrollbarRounding = 6.0f;
    style.TabRounding = 6.0f;
    style.WindowBorderSize = 0.0f;
    style.ChildBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.ItemSpacing = ImVec2(10, 8);
    style.ItemInnerSpacing = ImVec2(8, 6);
    style.WindowPadding = ImVec2(16, 14);
    style.FramePadding = ImVec2(10, 6);

    ImVec4 *colors = style.Colors;
    colors[ImGuiCol_WindowBg]             = ImVec4(0.07f, 0.08f, 0.10f, 1.0f);
    colors[ImGuiCol_ChildBg]              = ImVec4(0.10f, 0.11f, 0.14f, 1.0f);
    colors[ImGuiCol_PopupBg]              = ImVec4(0.12f, 0.13f, 0.17f, 0.98f);
    colors[ImGuiCol_Border]               = ImVec4(0.18f, 0.20f, 0.26f, 0.70f);
    colors[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]              = ImVec4(0.14f, 0.15f, 0.19f, 1.0f);
    colors[ImGuiCol_FrameBgHovered]       = ImVec4(0.19f, 0.21f, 0.28f, 1.0f);
    colors[ImGuiCol_FrameBgActive]        = ImVec4(0.24f, 0.27f, 0.36f, 1.0f);
    colors[ImGuiCol_TitleBg]              = ImVec4(0.08f, 0.09f, 0.11f, 1.0f);
    colors[ImGuiCol_TitleBgActive]        = ImVec4(0.12f, 0.14f, 0.18f, 1.0f);
    colors[ImGuiCol_MenuBarBg]            = ImVec4(0.09f, 0.10f, 0.13f, 1.0f);
    colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.08f, 0.09f, 0.11f, 0.5f);
    colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.22f, 0.25f, 0.32f, 1.0f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.30f, 0.34f, 0.44f, 1.0f);
    colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.38f, 0.43f, 0.55f, 1.0f);
    colors[ImGuiCol_CheckMark]            = ImVec4(0.38f, 0.68f, 1.00f, 1.0f);
    colors[ImGuiCol_SliderGrab]           = ImVec4(0.32f, 0.58f, 0.92f, 1.0f);
    colors[ImGuiCol_SliderGrabActive]     = ImVec4(0.42f, 0.70f, 1.00f, 1.0f);
    colors[ImGuiCol_Button]               = ImVec4(0.18f, 0.22f, 0.30f, 1.0f);
    colors[ImGuiCol_ButtonHovered]        = ImVec4(0.26f, 0.32f, 0.44f, 1.0f);
    colors[ImGuiCol_ButtonActive]         = ImVec4(0.32f, 0.40f, 0.56f, 1.0f);
    colors[ImGuiCol_Header]               = ImVec4(0.18f, 0.23f, 0.32f, 1.0f);
    colors[ImGuiCol_HeaderHovered]        = ImVec4(0.25f, 0.32f, 0.45f, 1.0f);
    colors[ImGuiCol_HeaderActive]         = ImVec4(0.30f, 0.38f, 0.54f, 1.0f);
    colors[ImGuiCol_Separator]            = ImVec4(0.18f, 0.20f, 0.26f, 0.60f);
    colors[ImGuiCol_SeparatorHovered]     = ImVec4(0.28f, 0.32f, 0.42f, 1.0f);
    colors[ImGuiCol_SeparatorActive]      = ImVec4(0.38f, 0.44f, 0.58f, 1.0f);
    colors[ImGuiCol_Text]                 = ImVec4(0.92f, 0.94f, 0.96f, 1.0f);
    colors[ImGuiCol_TextDisabled]         = ImVec4(0.50f, 0.54f, 0.62f, 1.0f);
    colors[ImGuiCol_PlotLines]            = ImVec4(0.38f, 0.68f, 1.00f, 1.0f);
    colors[ImGuiCol_PlotHistogram]        = ImVec4(0.32f, 0.75f, 0.55f, 1.0f);

    ImPlotStyle &plot_style = ImPlot::GetStyle();
    plot_style.Colors[ImPlotCol_FrameBg]       = ImVec4(0.08f, 0.09f, 0.12f, 1.0f);
    plot_style.Colors[ImPlotCol_PlotBg]        = ImVec4(0.08f, 0.09f, 0.12f, 1.0f);
    plot_style.Colors[ImPlotCol_PlotBorder]    = ImVec4(0.18f, 0.20f, 0.26f, 0.70f);
    plot_style.Colors[ImPlotCol_LegendBg]      = ImVec4(0.12f, 0.13f, 0.17f, 0.85f);
    plot_style.Colors[ImPlotCol_LegendBorder]  = ImVec4(0.22f, 0.25f, 0.32f, 0.80f);
    plot_style.Colors[ImPlotCol_Line]          = ImVec4(0.38f, 0.72f, 1.00f, 1.0f);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    NN::StudioEngine engine;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2((float)display_w, (float)display_h));
        ImGuiWindowFlags root_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
                                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

        ImGui::Begin("MainWorkspace", nullptr, root_flags);

        // Header sobrio: solo el estado del dispositivo
        if (engine.backend == NN::ExecutionBackend::GPU_Vulkan)
            ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.55f, 1.0f), "GPU (Vulkan): AMD Radeon RX 6800 XT");
        else if (engine.backend == NN::ExecutionBackend::CPU_Multithreaded)
            ImGui::TextColored(ImVec4(0.45f, 0.70f, 1.0f, 1.0f), "CPU: OpenMP Parallel");
        else
            ImGui::TextColored(ImVec4(0.70f, 0.75f, 0.85f, 1.0f), "CPU: Single Thread");

        ImGui::Separator();
        ImGui::Spacing();

        // Calcular con precisión la altura restante descontando header, padding y separadores
        ImVec2 content_avail = ImGui::GetContentRegionAvail();
        float content_height = content_avail.y;
        float sidebar_width = 390.0f;

        // ========================================================
        //  PANEL LATERAL: Configuración (Dataset, Capas, Backend)
        // ========================================================
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 12));
        ImGui::BeginChild("SidebarPanel", ImVec2(sidebar_width, content_height), true, ImGuiWindowFlags_None);

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.88f, 0.94f, 1.0f));
        ImGui::Text("DATASET");
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Spacing();

        if (engine.available_datasets.empty())
        {
            ImGui::TextDisabled("No hay datasets en 'datasets/'");
            if (ImGui::Button("Recargar", ImVec2(-1, 24)))
                engine.refreshDatasetList();
        }
        else
        {
            std::vector<std::string> ds_names;
            for (const auto &d : engine.available_datasets)
            {
                std::string label = d.folder_name + " (" + (d.kind == NN::DatasetKind::MNISTImages ? "MNIST" : "CSV") + ")";
                ds_names.push_back(label);
            }

            int cur = engine.selected_dataset_idx;
            std::vector<const char*> c_names;
            for (const auto &s : ds_names) c_names.push_back(s.c_str());

            ImGui::SetNextItemWidth(-1);
            if (ImGui::Combo("##DatasetCombo", &cur, c_names.data(), (int)c_names.size()))
            {
                engine.loadDataset(cur);
            }

            ImGui::Spacing();
            ImGui::TextDisabled("Muestras: %d | Entradas: %d | Clases: %d",
                                engine.current_dataset.num_samples,
                                engine.current_dataset.num_features,
                                engine.current_dataset.num_classes);
        }

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.88f, 0.94f, 1.0f));
        ImGui::Text("BACKEND & OPTIMIZER");
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Spacing();

        const char *backends[] = {"CPU (OpenMP Multithread)", "CPU (Single Thread)", "GPU (Vulkan Compute)"};
        int cur_bk = (engine.backend == NN::ExecutionBackend::CPU_Multithreaded) ? 0 : 
                     ((engine.backend == NN::ExecutionBackend::CPU_Singlethreaded) ? 1 : 2);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##BackendCombo", &cur_bk, backends, IM_ARRAYSIZE(backends)))
        {
            engine.backend = (cur_bk == 0) ? NN::ExecutionBackend::CPU_Multithreaded : 
                             ((cur_bk == 1) ? NN::ExecutionBackend::CPU_Singlethreaded : NN::ExecutionBackend::GPU_Vulkan);
        }

        const char *opt_names[] = {"Adam", "SGD + Momentum (0.9)", "SGD"};
        int cur_opt = (engine.optimizer == NN::Optimizer::Adam) ? 0 : 
                      ((engine.optimizer == NN::Optimizer::SGDMomentum) ? 1 : 2);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##OptimizerCombo", &cur_opt, opt_names, IM_ARRAYSIZE(opt_names)))
        {
            engine.optimizer = (cur_opt == 0) ? NN::Optimizer::Adam : 
                               ((cur_opt == 1) ? NN::Optimizer::SGDMomentum : NN::Optimizer::SGD);
            engine.invalidateModel();
        }

        const char *init_names[] = {
            "He Normal (Kaiming)",
            "He Uniform",
            "Glorot Uniform (Xavier)",
            "Glorot Normal",
            "LeCun Normal",
            "Random Uniform [-0.05, 0.05]"
        };
        int cur_init = static_cast<int>(engine.weight_init);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##InitCombo", &cur_init, init_names, IM_ARRAYSIZE(init_names)))
        {
            engine.weight_init = static_cast<NN::WeightInit>(cur_init);
            engine.invalidateModel();
        }

        ImGui::Spacing();
        ImGui::DragFloat("LR", &engine.learning_rate, 0.0001f, 0.00001f, 0.1f, "%.5f");
        ImGui::SliderInt("Épocas", &engine.epochs, 1, 100);
        ImGui::SliderInt("Batch Size", &engine.batch_size, 16, 256);

        ImGui::Spacing();
        if (engine.is_training.load())
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.70f, 0.20f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.25f, 0.25f, 1.0f));
            if (ImGui::Button("DETENER ENTRENAMIENTO", ImVec2(-1, 34)))
            {
                engine.stopTraining();
            }
            ImGui::PopStyleColor(2);
        }
        else
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.45f, 0.88f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.55f, 0.98f, 1.0f));
            if (ImGui::Button("ENTRENAR MODELO", ImVec2(-1, 34)))
            {
                engine.startTraining();
            }
            ImGui::PopStyleColor(2);
        }

        // ----------------------------------------------------
        // ARQUITECTURA DE CAPAS: Tabla compacta sin scrollbar fija
        // ----------------------------------------------------
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.88f, 0.94f, 1.0f));
        ImGui::Text("ARQUITECTURA");
        ImGui::PopStyleColor();
        ImGui::SameLine(sidebar_width - 85);
        if (ImGui::SmallButton("+ Capa"))
        {
            engine.layers.push_back({NN::LayerType::Dense, 64, NN::Activation::ReLU, 0.2f});
            engine.invalidateModel();
        }
        ImGui::Separator();
        ImGui::Spacing();

        if (engine.current_dataset.kind == NN::DatasetKind::MNISTImages)
            ImGui::TextColored(ImVec4(0.4f, 0.75f, 0.95f, 1.0f), "Input: [1 canal, 28x28] (784)");
        else
            ImGui::TextDisabled("Input: %d features", engine.current_dataset.num_features);

        // Contenedor de capas con auto-fit y sin scrollbar forzada
        ImGui::BeginChild("LayersScrollRegion", ImVec2(0, 0), false, ImGuiWindowFlags_None);
        for (int i = 0; i < (int)engine.layers.size(); ++i)
        {
            ImGui::PushID(i);
            auto &layer_item = engine.layers[i];

            // Barra compacta de cabecera de la capa
            const char *layer_type_names[] = {"Dense", "Dropout", "Conv2D", "MaxPooling2D", "Flatten", "Rescaling"};
            int cur_type_idx = static_cast<int>(layer_item.type);

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 3));
            ImGui::SetNextItemWidth(110);
            if (ImGui::Combo("##LayerType", &cur_type_idx, layer_type_names, IM_ARRAYSIZE(layer_type_names)))
            {
                layer_item.type = static_cast<NN::LayerType>(cur_type_idx);
                engine.invalidateModel();
            }

            ImGui::SameLine();
            if (layer_item.type == NN::LayerType::Dense)
            {
                ImGui::SetNextItemWidth(70);
                if (ImGui::DragInt("##Neurons", &layer_item.neurons, 1.0f, 4, 512, "%d n"))
                    engine.invalidateModel();

                ImGui::SameLine();
                const char *acts[] = {"ReLU", "Sigmoid", "Linear"};
                int act_idx = (layer_item.activation == NN::Activation::Sigmoid) ? 1 : 
                              ((layer_item.activation == NN::Activation::Linear) ? 2 : 0);
                ImGui::SetNextItemWidth(75);
                if (ImGui::Combo("##Act", &act_idx, acts, IM_ARRAYSIZE(acts)))
                {
                    layer_item.activation = (act_idx == 1) ? NN::Activation::Sigmoid : 
                                            ((act_idx == 2) ? NN::Activation::Linear : NN::Activation::ReLU);
                    engine.invalidateModel();
                }
            }
            else if (layer_item.type == NN::LayerType::Conv2D)
            {
                ImGui::SetNextItemWidth(65);
                if (ImGui::DragInt("##Filters", &layer_item.filters, 1.0f, 4, 64, "%d fil"))
                    engine.invalidateModel();
                ImGui::SameLine();
                ImGui::SetNextItemWidth(65);
                if (ImGui::DragInt("##Kernel", &layer_item.kernel_size, 1.0f, 3, 5, "%d k"))
                    engine.invalidateModel();
            }
            else if (layer_item.type == NN::LayerType::Dropout)
            {
                ImGui::SetNextItemWidth(140);
                if (ImGui::SliderFloat("##Rate", &layer_item.dropout_rate, 0.05f, 0.8f, "p: %.2f"))
                    engine.invalidateModel();
            }
            else if (layer_item.type == NN::LayerType::Rescaling)
            {
                ImGui::SetNextItemWidth(80);
                if (ImGui::DragFloat("##Scale", &layer_item.scale, 0.0001f, 0.00001f, 10.0f, "%.4f"))
                    engine.invalidateModel();
                ImGui::SameLine();
                if (ImGui::SmallButton("1/255"))
                {
                    layer_item.scale = 1.0f / 255.0f;
                    layer_item.offset = 0.0f;
                    engine.invalidateModel();
                }
            }
            else if (layer_item.type == NN::LayerType::MaxPooling2D)
            {
                ImGui::TextDisabled("2x2 s2");
            }
            else if (layer_item.type == NN::LayerType::Flatten)
            {
                ImGui::TextDisabled("-> 1D");
            }

            ImGui::SameLine(sidebar_width - 32);
            if (ImGui::SmallButton("X"))
            {
                engine.layers.erase(engine.layers.begin() + i);
                engine.invalidateModel();
                ImGui::PopStyleVar();
                ImGui::PopID();
                break;
            }
            ImGui::PopStyleVar();

            ImGui::PopID();
        }
        ImGui::EndChild(); // Fin LayersScrollRegion

        ImGui::EndChild(); // Fin SidebarPanel
        ImGui::PopStyleVar(); // Pop WindowPadding de Sidebar

        ImGui::SameLine();

        // ========================================================
        //  ÁREA PRINCIPAL: Pestañas de Trabajo (Tabs)
        // ========================================================
        float main_area_width = ImGui::GetContentRegionAvail().x;
        ImGui::BeginChild("MainAreaTabs", ImVec2(main_area_width, content_height), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        if (ImGui::BeginTabBar("WorkspaceTabs", ImGuiTabBarFlags_None))
        {
            // ----------------------------------------------------
            // TAB 1: Monitoreo & Gráficas de Entrenamiento
            // ----------------------------------------------------
            if (ImGui::BeginTabItem("  Entrenamiento & Métricas  "))
            {
                ImGui::Spacing();
                NN::LiveMetrics current_metrics;
                {
                    std::lock_guard<std::mutex> lk(engine.metrics_mtx);
                    current_metrics = engine.metrics;
                }

                if (engine.is_training.load())
                {
                    float prog = (current_metrics.total_epochs > 0) ? 
                                 (float)current_metrics.current_epoch / current_metrics.total_epochs : 0.0f;
                    ImGui::ProgressBar(prog, ImVec2(-1, 24), 
                        ("Época " + std::to_string(current_metrics.current_epoch) + " / " + std::to_string(current_metrics.total_epochs)).c_str());
                }

                ImGui::BeginChild("MetricsSummary", ImVec2(0, 52), true);
                ImGui::TextDisabled("Estado:"); ImGui::SameLine(); ImGui::Text("%s", current_metrics.status_msg.c_str());
                ImGui::SameLine(300);
                ImGui::TextDisabled("Test Accuracy:"); ImGui::SameLine(); 
                ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.55f, 1.0f), "%.2f%%", current_metrics.test_accuracy);
                ImGui::SameLine(500);
                ImGui::TextDisabled("Tiempo:"); ImGui::SameLine();
                ImGui::Text("%.1f ms", current_metrics.train_time_ms);
                ImGui::EndChild();

                ImGui::Spacing();
                float available_plot_h = ImGui::GetContentRegionAvail().y - 12.0f;
                if (ImPlot::BeginPlot("##LossPlot", ImVec2(-1, std::max(240.0f, available_plot_h))))
                {
                    ImPlot::SetupAxes("Época", "Loss");
                    if (!current_metrics.epoch_indices.empty())
                    {
                        ImPlot::PlotLine("Loss", current_metrics.epoch_indices.data(), 
                                         current_metrics.loss_history.data(), current_metrics.loss_history.size());
                    }
                    ImPlot::EndPlot();
                }

                ImGui::EndTabItem();
            }

            // ----------------------------------------------------
            // TAB 2: Inferencia Interactiva & Visualizador
            // ----------------------------------------------------
            if (ImGui::BeginTabItem("  Inferencia & Canvas  "))
            {
                ImGui::Spacing();
                if (engine.current_dataset.kind == NN::DatasetKind::MNISTImages)
                {
                    float cell_size = 7.5f;
                    float canvas_dim = 28.0f * cell_size;
                    ImVec2 canvas_size(canvas_dim, canvas_dim);

                    // Columna 1: Canvas de dibujo (sin BeginChild anidado)
                    ImGui::BeginGroup();
                    ImGui::TextDisabled("Lienzo de Entrada (28x28)");
                    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

                    ImDrawList *dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), IM_COL32(14, 16, 20, 255), 6.0f);
                    dl->AddRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), IM_COL32(50, 56, 72, 255), 6.0f);

                    ImGui::InvisibleButton("DrawingCanvas", canvas_size);
                    if (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left))
                    {
                        ImVec2 mpos = ImGui::GetIO().MousePos;
                        int cx = (int)((mpos.x - canvas_pos.x) / cell_size);
                        int cy = (int)((mpos.y - canvas_pos.y) / cell_size);

                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                int px = cx + dx, py = cy + dy;
                                if (px >= 0 && px < 28 && py >= 0 && py < 28)
                                {
                                    float dist = std::sqrt((float)(dx * dx + dy * dy));
                                    float intensity = (dist == 0.0f) ? 1.0f : 0.6f;
                                    float &pix = engine.drawing_canvas[py * 28 + px];
                                    pix = std::min(1.0f, pix + intensity);
                                }
                            }
                        }
                        if (engine.model_ready)
                            engine.predictCurrentInput();
                    }

                    for (int r = 0; r < 28; ++r)
                    {
                        for (int c = 0; c < 28; ++c)
                        {
                            float val = engine.drawing_canvas[r * 28 + c];
                            if (val > 0.01f)
                            {
                                ImVec2 p0(canvas_pos.x + c * cell_size, canvas_pos.y + r * cell_size);
                                ImVec2 p1(p0.x + cell_size, p0.y + cell_size);
                                dl->AddRectFilled(p0, p1, IM_COL32(255, 255, 255, (int)(val * 255)));
                            }
                        }
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("Borrar Lienzo", ImVec2(canvas_dim, 26)))
                        engine.clearCanvas();
                    ImGui::EndGroup();

                    ImGui::SameLine(0, 24.0f);

                    // Columna 2: Panel de probabilidades (con BeginGroup limpio sin scroll)
                    ImGui::BeginGroup();
                    ImGui::TextDisabled("Predicción");
                    if (engine.model_ready)
                        ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.55f, 1.0f), "%s", engine.predicted_class_name.c_str());
                    else
                        ImGui::TextDisabled("(Sin entrenar)");

                    ImGui::Spacing();
                    for (int d = 0; d < 10; ++d)
                    {
                        float p = (d < (int)engine.live_predictions.size()) ? engine.live_predictions[d] : 0.0f;
                        char buf[32];
                        snprintf(buf, sizeof(buf), "%d: %.1f%%", d, p * 100.0f);
                        ImGui::ProgressBar(p, ImVec2(220.0f, 16), buf);
                    }
                    ImGui::EndGroup();
                }
                else if (engine.current_dataset.kind == NN::DatasetKind::TabularCSV)
                {
                    ImGui::TextDisabled("Variables numéricas de entrada:");
                    int max_sliders = std::min(12, engine.current_dataset.num_features);
                    for (int f = 0; f < max_sliders; ++f)
                    {
                        std::string label = (f < (int)engine.current_dataset.feature_names.size()) ? 
                                             engine.current_dataset.feature_names[f] : ("Feature " + std::to_string(f));
                        if (ImGui::DragFloat(label.c_str(), &engine.generic_inputs[f], 0.5f))
                        {
                            if (engine.model_ready)
                                engine.predictCurrentInput();
                        }
                    }

                    ImGui::Spacing();
                    if (engine.model_ready)
                        ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.55f, 1.0f), "Predicción: %s", engine.predicted_class_name.c_str());
                    else
                        ImGui::TextDisabled("Predicción: (Modelo sin entrenar)");
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::TextDisabled("Visualización de Activaciones Neuronales");
                float visualizer_h = std::max(160.0f, ImGui::GetContentRegionAvail().y - 10.0f);
                RenderNetworkActivations(engine, ImVec2(ImGui::GetContentRegionAvail().x, visualizer_h));

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::EndChild(); // Fin MainAreaTabs
        ImGui::End();

        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    engine.stopTraining();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
