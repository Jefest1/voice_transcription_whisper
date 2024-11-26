import streamlit as st
import whisper
from IPython.display import display, Markdown
import tempfile
from transformers import pipeline
from dotenv import load_dotenv
import os
st.set_page_config(page_title="Voice Transcription", layout='wide')

# create a markdown for the page icons
st.markdown("""
    <style>
    .title {
        font-family: 'Arial', sans-serif;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("Voice transcription App using whisper")

load_dotenv()
# hf_token = os.getenv('HF_TOKEN')
hf_token = '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'
# Load the Whisper model once


@st.cache_resource
def load_whisper_model():
    return whisper.load_model('tiny')


model = load_whisper_model()


llama_model = "meta-llama/Llama-3.2-1B-Instruct"

# Function to transcribe audio


def transcribe_audio(file_path):
    try:
        result = model.transcribe(file_path)
        return result.get("text", "No transcription available")
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# function to get the transcriptions


def get_transcript(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(file.read())
            transcription = transcribe_audio(temp_audio.name)
            if transcription:
                return transcription
    except Exception as e:
        st.error(f"Error processing recorded audio: {e}")

# function to use llama to summarize the transcription


def summarize_transcription(message):
    try:
        pipe = pipeline(
            'text-generation',
            model=llama_model,
            device_map='auto',
            use_auth_token=hf_token
        )
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant for summarizing in markdown the text output generated by transcribing an audio'},
            {'role': 'user', 'content': message}
        ]
        response = pipe(messages)
        return response[0]['generated_text']
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return None

# function to get the user message


def get_user_message(transcription):
    user_message = f"Generate a summary for the text below using markdown format. The text is a transcribed text from another model\n\n{
        transcription}"
    return user_message


# Sidebar for file upload
with st.sidebar:
    st.header("Audio Upload")
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV, MP3)", type=["wav", "mp3"])
    st.header("Do you want AI summary of transcription?")
    get_ai_sum = st.checkbox("Get AI summary")

# Handle uploaded audio file
if uploaded_file:
    with st.spinner("Transcribing Audio..."):
        transcription = get_transcript(uploaded_file)
        st.subheader("Transcription for Uploaded Audio")
        st.write(transcription)
        # get the ai summary
    if get_ai_sum:
        try:
            user_message = get_user_message(transcription)
            response = summarize_transcription(user_message)
            st.header("AI summary")
            st.markdown(response)
        except Exception as e:
            st.error(f'Error Problem getting summary \n{e}')


# Audio input widget
audio_data = st.audio_input("Record your audio here")
if audio_data:
    transcription = get_transcript(audio_data)
    st.subheader("Transcription for Recorded Audio")
    st.write(transcription)
    if get_ai_sum:
        try:
            response = summarize_transcription(transcription)
            st.header("AI summary")
            st.markdown(response)
        except Exception as e:
            st.error(f'Error Problem getting summary \n{e}')
