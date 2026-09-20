import gradio as gr
from misaki import en, espeak
from kokoro_onnx import Kokoro
from pypdf import PdfReader

# load models into memory 
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

def generate_audio(text):
    # phonemize the text
    phonemes, _ = g2p(text)

    # generate audio samples
    samples, sample_rate = kokoro.create(phonemes, "af_heart", is_phonemes=True)

    return (sample_rate, samples)

def process_text(text):
    if not text:
        raise gr.Error("Please enter some text.")
    return generate_audio(text)

def process_file(file):
    if file is None:
        raise gr.Error("Please upload a file.")
        
    file_name = file.name
    final_text = ""
    
    if file_name.endswith(".txt"):
        with open (file_name, 'r', encoding='utf-8') as f:
            final_text = f.read()
    elif file_name.endswith(".pdf"):
        reader = PdfReader(file_name)    
        # pages_num = len(reader.pages)
        page = reader.pages[0]
        final_text = page.extract_text()
            
    if not final_text:
        raise gr.Error("Could not extract text from the file.")
        
    return generate_audio(final_text)

with gr.Blocks() as interface:
    gr.Markdown("# Text & File to Speech Generator")
    
    with gr.Tabs():

        with gr.TabItem("Type Text"):
            text_input = gr.Textbox(label="Input Text", lines=5, placeholder="Type text here...")
            text_btn = gr.Button("Generate from Text", variant="primary")
            

        with gr.TabItem("Upload File"):
            file_input = gr.File(label="Upload a .txt or .pdf file")
            file_btn = gr.Button("Generate from File", variant="primary")
            
   
    audio_output = gr.Audio(label="Generated Audio")
    

    text_btn.click(fn=process_text, inputs=text_input, outputs=audio_output, api_name="predict_text")
    file_btn.click(fn=process_file, inputs=file_input, outputs=audio_output, api_name="predict_file")

if __name__ == "__main__":
    interface.launch()