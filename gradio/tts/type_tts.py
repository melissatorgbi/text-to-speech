import gradio as gr
from misaki import en, espeak
from kokoro_onnx import Kokoro

# load models into memory 
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

def text_to_speech(text):
    # phonemize the text
    phonemes, _ = g2p(text)

    # generate audio samples
    samples, sample_rate = kokoro.create(phonemes, "af_heart", is_phonemes=True)

    return (sample_rate, samples)

interface = gr.Interface(
    fn=text_to_speech,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Audio(label="Generated Audio"),
    api_name="predict"
)

interface.launch()