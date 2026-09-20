# Text-to-Speech (TTS) Project

This repo contains a Text-to-Speech (TTS) application built around the Kokoro neural TTS model using the `kokoro-onnx` library.

## 📁 Project Structure

```
├── examples/                
├── gradio/ 
│   └── example/
│   └── tts/
├── Readme.md                       
└── requirements.txt   

```

The `examples/` directory contains starter examples sourced from the official [kokoro-onnx repo](https://github.com/thewh1teagle/kokoro-onnx). 
The `gradio/` directory contains both examples of Gradio applications and several varying TTS application interface that take text or a file and output audio.

## 🚀 Getting Started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation):

```console
pip install uv
```

2. Run in the following:

```console
uv init -p 3.12
uv add kokoro-onnx soundfile
```
3. Download model files

Download [kokoro-v1.0.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx) and [voices-v1.0.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin) and place them in your working directory.

## 💻 Usage
### Gradio
The `gradio/` directory contains the main interactive project. Running the application scripts will launch a local web server with a UI for inputting text and listening to generated audio. For example:

```console
uv run gradio/tts/type_tts.py
```

More information on Gradio [here](https://gradio.app/).