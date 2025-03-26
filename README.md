# Speech2Text Whisper Gradio App  
**Multilingual Transcription & Translation with Whisper + Gradio**

Effortlessly transcribe and translate audio files locally using [Whisper models](https://huggingface.co/openai) from OpenAI. Built with [Gradio](https://www.gradio.app/), a browser-based interface. This app runs locally - your data stays with you.

---

## Features

- Transcribe audio to native language text  
- Translate audio directly into English  
- Model selection from [OpenAI Whisper](https://huggingface.co/openai) variants  
- Powered by [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  
- Fully local processing = Maximum privacy  
- Sleek UI with [Gradio Blocks](https://www.gradio.app/docs/blocks/)  
- Batch processing with time metrics  
- Downloadable transcriptions in `.txt` format  

---

## Why Use This?

Use Cases:

- Journalists recording interviews  
- Podcasters needing quick transcripts  
- Law enforcement & legal professionals documenting field audio  
- Researchers analyzing voice recordings
- Researchers working in a lab, recording experiments and protocols for documenting later  
- Teachers recording multilingual classroom sessions  
- Creators generating subtitles for videos  

Data Security & Privacy:  
This app is fully local - your audio files never leave your machine. No uploads, no external API calls. You are in total control of your data.

---

## Live Demo Screenshot

Below is a screenshot example of the app in action:

![Speech2Text Whisper App Screenshot](/screenshot/Speech2TextWhisperGradioApp.png)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/speech2text-whisper-app.git
cd speech2text-whisper-app
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```
Or use conda to create environment 

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gradio transformers pydub numpy
```

### 4. Install FFmpeg
This app uses [`pydub`](https://github.com/jiaaro/pydub), which requires FFmpeg to handle various audio formats.

- Windows: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html), and add it to your system PATH.
- macOS (with Homebrew):
```bash
brew install ffmpeg
```
- Linux (Debian/Ubuntu):
```bash
sudo apt update
sudo apt install ffmpeg
```

---

## Run the App

```bash
python Speech2TextWhisperGradioApp.py
```

This will open a browser window at `http://127.0.0.1:7860`. Upload audio files (any format), select a model, and click "Transcribe".

---

## Supported Whisper Models

Choose from the following Whisper models. VRAM requirements are approximate and depend on system configuration and batch size.

| Model                         | Parameters | VRAM Requirement | Performance               |
|------------------------------|------------|------------------|---------------------------|
| `openai/whisper-small`       | 244M       | ~2 GB            | Fast, less accurate       |
| `openai/whisper-medium`      | 769M       | ~5 GB            | Balanced                  |
| `openai/whisper-large`       | 1.55B      | ~10 GB           | More accurate             |
| `openai/whisper-large-v3`    | 1.55B      | ~10 GB           | Most accurate             |
| `openai/whisper-large-v3-turbo` | 809M    | ~6 GB            | Optimized for speed       |

Reference: [Whisper GitHub Repository](https://github.com/openai/whisper)

---

## File Output

For each audio file, you will get:

- Native language transcription: `filename_native.txt`
- English translation: `filename_english.txt`

Both files are downloadable directly from the app interface.

---

## Tips

- Clear audio produces better results  
- Rename files before uploading to stay organized  
- Larger models are more accurate but require more memory  
- No internet required - runs offline  

---

If you find this project helpful:

- Star it on GitHub  
- Share it on social platforms with screenshots and your use-case  
- Fork and extend it into your own transcription product  
- Leave a comment or suggestion in the Issues tab  

---

## Improvements/Features ongoing

- Speaker diarization and segmentation, read these:
- https://hf.co/pyannote/speaker-diarization
- https://hf.co/pyannote/segmentation

---

## Contact

For queries, feature requests, or collaborations:

Email: [drsanjivk@gmail.com](mailto:drsanjivk@gmail.com)  
GitHub: [github.com/sanjiv856](https://github.com/sanjiv856)

---

## License

© 2025 Sanjiv Kumar  
This project is licensed under the [MIT License](LICENSE).

---
