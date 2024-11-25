import streamlit as st
import whisper
from IPython.display import display, Markdown
import tempfile
from langchain_community.llms import Ollama

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

# Load the Whisper model once


@st.cache_resource
def load_whisper_model():
    return whisper.load_model('tiny')

# Load the llama 3.2 model once


@st.cache_resource
def load_llama_():
    return Ollama(model='llama3.2')


model = load_whisper_model()
llama_model = load_llama_()
# Function to transcribe audio


def transcribe_audio(file_path):
    try:
        result = model.transcribe(file_path)
        return result.get("text", "No transcription available")
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# function to get the transcripts


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
    message = f'Present the summary of the text transcribed from an audio in markdown which is below\n{
        message}'
    reponse = llama_model.invoke(message)
    return reponse


# Sidebar for file upload
with st.sidebar:
    st.header("Audio Upload")
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV, MP3)", type=["wav", "mp3"])
    st.header("Do you want AI summary of transcription?")
    get_ai_sum = st.checkbox("Get AI summary")

# Handle uploaded audio file
if uploaded_file:
    transcription = get_transcript(uploaded_file)
    st.subheader("Transcription for Uploaded Audio")
    st.write(transcription)
    # get the ai summary
    if get_ai_sum:
        try:
            response = summarize_transcription(transcription)
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
