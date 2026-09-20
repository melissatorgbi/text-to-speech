import gradio as gr
import asyncio
import sounddevice as sd
from kokoro_onnx import Kokoro

# load model into memory 
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

def text_to_speech(text):
    async def text_to_speech_stream():
        
        stream = kokoro.create_stream(
            text,
            voice="af_nicole",
            speed=1.0,
            lang="en-us",
        )

        count = 0
        async for samples, sample_rate in stream:
            count += 1
            print(f"Playing audio stream ({count})...")
            sd.play(samples, sample_rate)
            sd.wait()


    asyncio.run(text_to_speech_stream())

interface = gr.Interface(
    fn=text_to_speech,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Audio(label="Generated Audio"),
    api_name="predict"
)

interface.launch()