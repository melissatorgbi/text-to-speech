import gradio as gr
from misaki import en, espeak
from kokoro_onnx import Kokoro
from pypdf import PdfReader

# load models into memory 
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

def file_to_speech(file):
    
    file_name = file.name

    if file_name.endswith(".txt"):
        with open (file_name, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file_name.endswith(".pdf"):
        reader = PdfReader(file_name)    
        # pages_num = len(reader.pages)
        page = reader.pages[0]
        text = page.extract_text()


    # phonemize the text
    phonemes, _ = g2p(text)

    # generate audio samples
    samples, sample_rate = kokoro.create(phonemes, "af_heart", is_phonemes=True)

    return (sample_rate, samples)

interface = gr.Interface(
    fn=file_to_speech,
    inputs=gr.File(),
    outputs=gr.Audio(label="Generated Audio"),
    api_name="predict"
)

interface.launch()    
