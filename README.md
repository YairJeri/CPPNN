# Red Neuronal Acelerada con Vulkan en C++

Este proyecto es una implementación modular de una red neuronal en C++ con soporte para aceleración en GPU mediante Vulkan. No depende de frameworks de machine learning como TensorFlow o PyTorch.

---

## Características

- Implementación desde cero de redes feedforward con backpropagation.
- Tensores de hasta 4 dimensiones (`float32`).
- Shaders de cómputo escritos en GLSL y compilados a SPIR-V.
- Aceleración en GPU mediante Vulkan Compute.
- Arquitectura extensible con sistema de capas (`Dense`, `Conv2D`, `Flatten`, etc.).
- Separación entre lógica de CPU y ejecución en GPU.

---

## Dependencias

- [Vulkan SDK](https://vulkan.lunarg.com/)
- [glslangValidator](https://github.com/KhronosGroup/glslang) (incluido en el Vulkan SDK)
- CMake 3.15 o superior
- C++17 o superior

> Nota: No se utilizan bibliotecas de alto nivel ni frameworks de machine learning.

---

## En desarrollo

- [ ] Capas convolucionales (`Conv2D`, `MaxPooling`, etc.).
- [ ] Entrenamiento completamente en GPU.
- [ ] Visualización de arquitectura y resultados.
- [ ] Serialización y carga de modelos entrenados.

---

## Compilación

1. Asegúrate de tener el Vulkan SDK instalado y la variable de entorno `VULKAN_SDK` configurada correctamente.
2. Clona el repositorio.
3. Ejecuta los siguientes comandos:

```bash
mkdir build
cd build
cmake ..
make
```
