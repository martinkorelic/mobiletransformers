# 📱 ORTransformersMobile: An On-Device LLM Framework for Fine-Tuning and Inference

**ORTransformersMobile** is a modular framework designed for fully **on-device execution** of large and small language models (LLM / SLM) on mobile and edge devices.  
Built on top of **ONNX Runtime**, it leverages hardware-accelerated execution providers such as **XNNPACK**, **NNAPI**, and **QNN** for efficient inference and training on Android and similar platforms.

- **OR**: ONNX Runtime  
- **Transformers**: Core architecture of large language models  
- **Mobile**: Fully on-device mobile execution  

---

## 🚀 What is ORTransformersMobile?

A comprehensive, privacy-first framework that empowers researchers and developers to export, fine-tune, merge, and deploy transformer-based language models directly on your Android device. Eliminate dependency on cloud services while maintaining full control over your AI models in your pocket.
Perfect for privacy-preserving NLP applications, offline AI assistants, personalized chatbots, and edge computing scenarios where data sovereignty and real-time responsiveness are crucial. Whether you're building the next generation of pocket AI or developing enterprise edge solutions, **ORTransformersMobile** provides the foundation for truly autonomous mobile intelligence.

**Key Benefits**:

- 🔒 **Complete Privacy**: Your data never leaves your device
- 📱 **Pocket-Sized AI**: Full LLM/SLM capabilities in your smartphone
- 🔧 **Hardware execution provider support**: Hardware-accelerated inference for efficient on-device execution
- 🌐 **Offline-First**: Works anywhere, anytime, without internet connectivity
- 🤖 **Universal Model Support**: Compatible with most custom LLMs/SLMs from Huggingface

---

## 📦 Repository Contents

This comprehensive repository provides everything needed for on-device LLM deployment:

- 🔄 **Export Pipeline**: Streamlined conversion system transforming Huggingface LLMs/SLMs into PEFT-enabled training models and ONNX inference graphs optimized for Android deployment
- 📱 **Complete Android Application**: Full-featured Android folder containing the entire mobile application stack, ready for pocket deployment
- 🧪 **Custom PEFT support**: Customizable PEFT solutions for on-device fine-tuning (e.g. LoRA - Low-rank approximation, MARS - Multi-Adapter Rank Sharing and more)
- 🐍 **Training & Inference Scripts**: Python implementations supporting both PyTorch and ONNX Runtime, optimized for mobile hardware constraints
- 🔬 **Evaluation Scripts**: Comprehensive benchmarking suite for trained models across diverse NLP tasks, including mobile-specific performance metrics and battery consumption analysis

---

## 📱 Android Application: ORTransformer-android

The Android app is split into two main parts:

- 📲 **Kotlin UI Layer**  
  A lightweight interface acting as a communication bridge, calling APIs from the backend on the mobile device

- ⚙️ **Backend: ORTransformersMobile**  
  The core engine of the entire framework, implemented in **Kotlin and C++**. Can be easily implemented in re-used in another application, pick and choose which features you need.
  🔧 Key features include:  
  - **Modular Android Project**: Clean separation of concerns with isolated modules for **training**, **inference**, **RAG / CAG** and **weight management** 
  - **Hardware-Accelerated Loops**: **On-device training / fine-tuning** and generation loops leveraging NNAPI, XNNPACK, and Qualcomm QNN for optimal mobile performance
  - **Dynamic Configuration**: Real-time customization of training parameters and inference settings tailored to your Android device's capabilities
  - **ONNX Runtime Integration**: Optimized model execution specifically tuned for mobile and edge hardware 
  - **Weight Management**: **On-device weight merging** with automatic export to **Android filesystem**, enabling model personalization without cloud dependency
  - **Seamless Model Loading**: Direct import of merged weights into inference graphs for immediate pocket deployment
  - **RAG support**: Support for **Retrieval-Augmented Generation (RAG)** using **ObjectBox C++** as a lightning-fast **on-device vector database**
  - **CAG support**: Support for **Cached-Augmented Generation (CAG)** utilizing KV cache and embedding tensors during generation  


---

## ✅ Key Capabilities

| Feature                                     | Description                                                        |
|---------------------------------------------|------------------------------------------------------------------|
| ✅ Export **custom PyTorch Huggingface SLM / LLM models** | Convert Huggingface models with PEFT methods to training & ONNX inference models for on-device use |
| ✅ On-device **fine-tuning/training** loop       | Perform parameter-efficient training (PEFT) directly on mobile devices |
| ✅ On-device **generation** loop with KV caching | Efficient text generation using cached key-value tensors for faster autoregressive inference |
| ✅ **Customizable** training and generation      | Flexible configuration to adapt training and generation to specific tasks and hardware |
| ✅ On-device **weight exporting**                 | Save trained or merged weights directly on-device (mobile filesystem) |
| ✅ On-device **weight merging**                    | Merge base and PEFT weights on-device, with optional quantization for optimized size and speed |
| ✅ Direct inference from **merged weights**       | Load merged weights into the inference graph for seamless on-device model execution |
| ✅ **Retrieval-Augmented Generation** (RAG)       | Fully on-device vector database integration with ObjectBox for augmented generation |
| ✅ **Cached-Augmented Generation** (CAG)          | Leverage KV cache and embedding reuse during generation to accelerate inference |

---

## 🛠️ Built On

- [**ONNX Runtime**](https://onnxruntime.ai/) for training/inference and support for mobile-optimized execution providers:  
  - XNNPACK  
  - NNAPI  
  - Qualcomm QNN  
- [**Huggingface Transformers**](https://huggingface.co/) ecosystem compatibility for model export  
- [**ObjectBox C++**](https://objectbox.io/) for lightweight on-device vector databases in RAG workflows  

---

## 🎯 Why ORTransformersMobile?

- Fully **on-device** - no cloud dependency, maximizing privacy and minimizing latency 
- Enables **parameter-efficient fine-tuning (PEFT)** on mobile hardware  
- Modular and customizable for research and production use
- Ready for **Android** and adaptable to other edge devices  
- Combines cutting-edge generation techniques with practical on-device deployment  

---

## 🔧 Extensibility and Future Work

ORTransformersMobile is designed as a flexible platform, allowing easy extension for advanced on-device ML workflows, such as:

- Beyond text generation - classification, sentiment analysis, named entity recognition, question answering, summarization, and custom NLP tasks tailored for mobile use cases
- On-device **reinforcement learning**  
- **Federated learning** leveraging exported merged weights  
- Integration with additional hardware acceleration backends  
- Support for more PEFT methods and quantization techniques  
- Expansion to other mobile platforms and edge systems  

---

## 📥 Getting Started

*Coming soon:*  
Detailed installation instructions, export guides, training and inference examples, and API documentation.

---

## Citation

If you are using this framework for your own work, please cite:

```
@misc{ortransformersmobile2025,
  author       = {Koreli\v{c}, Martin and Pejovi{\'c}, Veljko},
  title        = {ORTransformersMobile: An On-Device LLM Framework for Fine-Tuning and Inference},
  year         = {2025},
  howpublished = {\url{https://gitlab.fri.uni-lj.si/lrk/ortransformersmobile}},
  note         = {ORTransformersMobile is a lightweight, modular framework for running and adapting large language models (LLMs) directly on mobile and edge devices. It supports on-device fine-tuning (PEFT), efficient inference using ONNX Runtime, on-device weight merging and quantization, and direct inference from merged weights. Advanced generation techniques include Retrieval-Augmented Generation (RAG) with vector databases and Cached-Augmented Generation (CAG) leveraging KV-cache and embedding reuse. Accessed: 2025-08-05}
}
```

---

## Acknowledgements

This work was supported by the Slovenian Research Agency grant no. N2-0393 approXimation for adaptable diStributed artificial intelligence and grant no. J2-3047 Context-Aware On-Device Approximate Computing.

---