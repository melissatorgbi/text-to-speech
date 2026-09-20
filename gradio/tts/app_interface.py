import gradio as gr
from misaki import en, espeak
from kokoro_onnx import Kokoro
from pypdf import PdfReader

# load models into memory 
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

def convert_to_speech(text, file):
    final_text = ""
    
    if text:
        final_text = text
    elif file is not None:
        file_name = file.name
        
        if file_name.endswith(".txt"):
            with open (file_name, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_name.endswith(".pdf"):
            reader = PdfReader(file_name)    
            # pages_num = len(reader.pages)
            page = reader.pages[0]
            text = page.extract_text()
                
    if not final_text:
        raise gr.Error("Please enter text or upload a valid .txt or .pdf file.")

    # phonemize the text
    phonemes, _ = g2p(final_text)

    # generate audio samples
    samples, sample_rate = kokoro.create(phonemes, "af_heart", is_phonemes=True)

    return (sample_rate, samples)

interface = gr.Interface(
    fn=convert_to_speech,
    inputs=[
        gr.Textbox(label="Input Text", lines=5, placeholder="Type text here..."),
        gr.File(label="Or upload a file")
    ],
    outputs=gr.Audio(label="Generated Audio"),
    # title="",
    api_name="predict"
)

if __name__ == "__main__":
    interface.launch()