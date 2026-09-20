import gradio as gr
from misaki import en, espeak
from kokoro_onnx import Kokoro
from pypdf import PdfReader

# load models into memory 
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

def convert_to_speech(text_input, file_input, start_page, end_page):
    final_text = text_input or ""
    
    if not final_text and file_input is not None:
        file_name = file_input.name
        
        if file_name.endswith(".txt"):
            with open(file_name, 'r', encoding='utf-8') as f:
                final_text = f.read()
        elif file_name.endswith(".pdf"):
            reader = PdfReader(file_name)
            total_pages = len(reader.pages)
            
            if start_page < 1 or end_page > total_pages or start_page > end_page:
                raise gr.Error(f"Page range error. The uploaded PDF has {total_pages} pages. Ensure your start and end pages are valid.")
            
            selected_pages = reader.pages[int(start_page)-1 : int(end_page)]
            final_text = "\n".join([p.extract_text() for p in selected_pages if p.extract_text()])
                
    if not final_text.strip():
        raise gr.Error("Please provide text or a valid file containing extractable text.")

    phonemes, _ = g2p(final_text)
    samples, sample_rate = kokoro.create(phonemes, "af_heart", is_phonemes=True)

    return (sample_rate, samples)

interface = gr.Interface(
    fn=convert_to_speech,
    inputs=[
        gr.Textbox(label="Input Text"),
        gr.File(label="Upload File"),
        gr.Number(label="Start Page", value=1, precision=0),
        gr.Number(label="End Page", value=1, precision=0)
    ],
    outputs=gr.Audio(),
    api_name="predict"
)

if __name__ == "__main__":
    interface.launch()