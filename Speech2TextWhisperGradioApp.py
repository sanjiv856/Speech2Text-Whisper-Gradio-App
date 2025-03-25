import os
import time
import numpy as np
import torch
import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16

# Global cache
pipeline_cache = {}

def load_pipeline(model_id):
    if model_id in pipeline_cache:
        return pipeline_cache[model_id]

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )
    pipeline_cache[model_id] = pipe
    return pipe

def convert_audio_to_mp3(input_file):
    ext = os.path.splitext(input_file)[1].lower()
    if ext == ".mp3":
        return input_file
    output_file = os.path.splitext(input_file)[0] + ".mp3"
    audio = AudioSegment.from_file(input_file, format=ext.replace(".", ""))
    audio.export(output_file, format="mp3", bitrate="192k")
    return output_file

def save_transcription(audio_file, text, lang):
    base_name = os.path.splitext(audio_file)[0]
    text_file = f"{base_name}_{lang}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)
    return text_file

def transcribe_audio_files(audio_files, model_choice):
    start_time = time.time()
    pipe = load_pipeline(model_choice)

    native_transcriptions = []
    english_transcriptions = []
    native_files = []
    english_files = []
    file_names = []

    for audio_file in audio_files:
        mp3_file = convert_audio_to_mp3(audio_file)
        audio = AudioSegment.from_file(mp3_file, format="mp3").set_channels(1).set_frame_rate(16000)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        samples = samples.squeeze()
        audio_list = [samples]

        # Native transcription
        native_result = pipe(audio_list, return_timestamps=True, chunk_length_s=30, generate_kwargs={"task": "transcribe"})
        if isinstance(native_result, list):
            native_result = native_result[0]
        native_text = native_result.get("text", "")
        native_transcriptions.append(f"{os.path.basename(audio_file)}:\n{native_text}\n")
        native_file = save_transcription(audio_file, native_text, "native")
        native_files.append(native_file)

        # English translation
        english_result = pipe(audio_list, return_timestamps=True, chunk_length_s=30, generate_kwargs={"task": "translate", "language": "english"})
        if isinstance(english_result, list):
            english_result = english_result[0]
        english_text = english_result.get("text", "")
        english_transcriptions.append(f"{os.path.basename(audio_file)}:\n{english_text}\n")
        english_file = save_transcription(audio_file, english_text, "english")
        english_files.append(english_file)

        file_names.append(os.path.basename(audio_file))

    total_time = time.time() - start_time
    file_list_display = "\n".join(file_names)
    processing_info = f"Processed {len(audio_files)} file(s) in {total_time:.2f} seconds."

    return (
        file_list_display,
        "\n".join(native_transcriptions),
        "\n".join(english_transcriptions),
        native_files,
        english_files,
        processing_info
    )

# Choices for models
MODEL_CHOICES = [
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo"
]

# Gradio Interface
with gr.Blocks(theme="ParityError/Interstellar") as app:
    gr.Markdown("# Speech2Text Whisper Audio App")
    gr.Markdown("Upload audio files to get transcription in the native language and English translation.")

    with gr.Row():
        audio_input = gr.Files(label="Upload Audio Files", type="filepath")
        with gr.Column():
            model_selector = gr.Dropdown(label="Select Whisper Model", choices=MODEL_CHOICES, value=MODEL_CHOICES[3])
            transcribe_button = gr.Button("Transcribe")

    file_name_output = gr.Textbox(label="Input File Names", interactive=False)

    with gr.Row():
        native_output = gr.Textbox(label="Native Language Transcription", lines=10, interactive=False)
        english_output = gr.Textbox(label="English Translation", lines=10, interactive=False)

    with gr.Row():
        native_files_output = gr.File(label="Download Native Transcriptions")
        english_files_output = gr.File(label="Download English Translations")

    time_display = gr.Textbox(label="Time Taken", interactive=False)

    transcribe_button.click(
        transcribe_audio_files,
        inputs=[audio_input, model_selector],
        outputs=[file_name_output, native_output, english_output, native_files_output, english_files_output, time_display]
    )

    # 📞 Contact Info
    gr.Markdown("## Contact ")
    gr.Markdown("""
    **If you experience any issues with the app, please contact us:**  
    - **Email:** [drsanjivk@gmail.com](mailto:drsanjivk@gmail.com)
    """)

    # © Footer
    gr.Markdown("#### © 2025 Sanjiv Kumar. All Rights Reserved.")

app.launch()
