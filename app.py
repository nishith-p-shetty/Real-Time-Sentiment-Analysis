import os
import gradio as gr
import whisper
from transformers import pipeline

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Load the Whisper ASR model
model = whisper.load_model("base")

# Load the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions")

# Function to analyze sentiment
def analyze_sentiment(text):
    try:
        results = sentiment_analysis(text)
        sentiment_results = {result['label']: result['score'] for result in results}
        return sentiment_results
    except Exception as e:
        return {"error": str(e)}

# Function to get corresponding emoji for a sentiment
def get_sentiment_emoji(sentiment):
    emoji_mapping = {
        "disappointment": "😞",
        "sadness": "😢",
        "annoyance": "😠",
        "neutral": "😐",
        "disapproval": "👎",
        "realization": "😮",
        "nervousness": "😬",
        "approval": "👍",
        "joy": "😄",
        "anger": "😡",
        "embarrassment": "😳",
        "caring": "🤗",
        "remorse": "😔",
        "disgust": "🤢",
        "grief": "😥",
        "confusion": "😕",
        "relief": "😌",
        "desire": "😍",
        "admiration": "😌",
        "optimism": "😊",
        "fear": "😨",
        "love": "❤️",
        "excitement": "🎉",
        "curiosity": "🤔",
        "amusement": "😄",
        "surprise": "😲",
        "gratitude": "🙏",
        "pride": "🦁"
    }
    return emoji_mapping.get(sentiment, "")

# Function to display sentiment analysis results
def display_sentiment_results(sentiment_results):
    if "error" in sentiment_results:
        return f"Error: {sentiment_results['error']}"
    
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        sentiment_text += f"{sentiment} {emoji}: {score:.2f}\n"
    return sentiment_text

# Main function to process the audio input
def inference(audio_path):
    try:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)

        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        sentiment_results = analyze_sentiment(result.text)
        sentiment_output = display_sentiment_results(sentiment_results)

        return lang.upper(), result.text, sentiment_output
    except Exception as e:
        return "Error", "", str(e)

# Gradio interface setup
title = """<h1 align="center">🎤 Multilingual Sentiment Analysis 💬</h1>"""
image_path = "thmbnail.jpg"
description = """
💻 This demo showcases a general-purpose speech recognition model called Whisper. It is trained on a large dataset of diverse audio and supports multilingual speech recognition, speech translation, and language identification tasks.
<br>
<br>
⚙️ Components of the tool:
<br>
&nbsp;&nbsp;&nbsp;&nbsp; - Real-time multilingual speech recognition<br>
&nbsp;&nbsp;&nbsp;&nbsp; - Language identification<br>
&nbsp;&nbsp;&nbsp;&nbsp; - Sentiment analysis of the transcriptions<br>
<br>
🎯 The sentiment analysis results are provided as a dictionary with different emotions and their corresponding scores.<br>
<br>
😃 The sentiment analysis results are displayed with emojis representing the corresponding sentiment.<br>
<br>
✅ The higher the score for a specific emotion, the stronger the presence of that emotion in the transcribed text.<br>
<br>
❓ Use the microphone for real-time speech recognition.<br>
<br>
⚡️ The model will transcribe the audio and perform sentiment analysis on the transcribed text.<br>
<br>
"""

custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""

# Gradio app setup
demo = gr.Blocks(css=custom_css)

with demo:
    gr.HTML(title)

    with gr.Row():
        with gr.Column():
            gr.HTML(description)
        with gr.Column():
            gr.Image(image_path, elem_id="banner-image", show_label=False)

    with gr.Group():
        with gr.Row():
            audio = gr.Audio(label="Input Audio", type="filepath")

            with gr.Column():
                btn = gr.Button("Transcribe", size="lg", variant="primary")
                lang_str = gr.Textbox(label="Language", interactive=False)
                text = gr.Textbox(label="Transcription", interactive=False)
                sentiment_output = gr.Textbox(label="Sentiment Analysis Results", interactive=False)

    btn.click(inference, inputs=[audio], outputs=[lang_str, text, sentiment_output])

# Launch the demo
demo.launch(share=False)
