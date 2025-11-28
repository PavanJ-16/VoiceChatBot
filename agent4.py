import streamlit as st
import requests
import base64
import os
from openai import OpenAI
from st_audiorec import st_audiorec
import dotenv
# Custom CSS for aesthetic enhancements
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
.stButton > button {
    background: linear-gradient(45deg, #1f77b4, #4a90e2);
    color: white;
    border-radius: 20px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stAudio {
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.chat-bubble {
    background: white;
    padding: 1rem;
    border-radius: 18px;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.user-bubble {
    background: linear-gradient(45deg, #4a90e2, #7b68ee);
    color: white;
    text-align: right;
}
.bot-bubble {
    background: #f0f2f6;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# Load API keys from env vars with sidebar fallback
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")
sarvam_api_key = os.getenv("SARVAM_API_KEY")  or st.sidebar.text_input("Sarvam AI API Key", type="password")
gemini_api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not sarvam_api_key or not gemini_api_key:
    st.sidebar.warning("👆 Enter your API keys to start.")
    st.stop()

# Initialize OpenAI client for Gemini
openai_client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# STT function (Sarvam)
@st.cache_data
def speech_to_text(audio_bytes):
    url = "https://api.sarvam.ai/speech-to-text"
    files = {'file': ('recorded.wav', audio_bytes, 'audio/wav')}
    data = {
        'model': 'saarika:v2.5',
        'language_code': 'unknown',  # Auto-detect
        'with_timestamps': 'false',
        'with_diarization': 'false'
    }
    headers = {'api-subscription-key': sarvam_api_key}
    response = requests.post(url, files=files, data=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        transcript = result.get('transcript', '')
        lang = result.get('language_code', 'en-IN')
        return transcript, lang
    else:
        st.error(f"STT Error: {response.text}")
        return "", "en-IN"

# LLM function (Gemini via OpenAI client) - Concise prompt
def get_llm_response(user_message, lang="en-IN"):
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Respond conversationally, matching the detected language context ({lang}). Use natural code-mixing if appropriate. Keep responses under 400 characters for voice output."}
    ]
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    response = openai_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        temperature=0.7,
        stream=False
    )
    ai_message = response.choices[0].message.content
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": ai_message})
    return ai_message

# TTS function (Sarvam) - With truncation
@st.cache_data
def text_to_speech(text, lang="en-IN"):
    # Truncate if exceeds 500 chars
    if len(text) > 500:
        text = text[:500] + "..."
        st.warning("Response truncated for voice output (Sarvam limit: 500 chars).")
    
    url = "https://api.sarvam.ai/text-to-speech"
    payload = {
        "inputs": [text],
        "target_language_code": lang,
        "speaker": "anushka",
        "model": "bulbul:v2",
        "pace": 1.0,
        "pitch": 0.0,
        "loudness": 1.0,
        "speech_sample_rate": "22050",
        "enable_preprocessing": False
    }
    headers = {
        "api-subscription-key": sarvam_api_key,
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        audio_b64 = result.get('audios', [''])[0]
        audio_bytes = base64.b64decode(audio_b64)
        return audio_bytes
    else:
        st.error(f"TTS Error: {response.text}")
        return None

# Main UI - Aesthetic Header
st.markdown("# 🗣️✨ Voice-to-Voice Multilingual Chatbot")
st.markdown("**Powered by Sarvam AI (STT/TTS) & Gemini (LLM)** – Speak naturally in Hindi, Tamil, English, and more!")
st.markdown("---")

# Audio recorder - Centered with markdown label (no unsupported kwargs)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### 🎤 **Click to Record Your Voice**")
    wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # User input preview in column
    col_u, col_r = st.columns(2)
    with col_u:
        st.markdown('<div class="chat-bubble user-bubble">**You said:**</div>', unsafe_allow_html=True)
        st.audio(wav_audio_data, format='audio/wav')
    
    with st.spinner("🔍 Transcribing..."):
        transcript, detected_lang = speech_to_text(wav_audio_data)
    
    if transcript:
        st.markdown(f"**Detected Language:** {detected_lang}")
        
        with st.spinner("🤖 Generating response..."):
            response_text = get_llm_response(transcript, detected_lang)
        
        # Response in columns
        with col_r:
            st.markdown(f'<div class="chat-bubble bot-bubble">**Bot:** {response_text}</div>', unsafe_allow_html=True)
        
        with st.spinner("🔊 Synthesizing voice..."):
            audio_out = text_to_speech(response_text, detected_lang)
            if audio_out:
                st.audio(audio_out, format='audio/wav', autoplay=True)
            else:
                st.warning("❌ Failed to generate audio.")

# Chat history - In expander for aesthetics
with st.expander("💬 Chat History", expanded=False):
    for msg in st.session_state.messages:
        role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        st.markdown(f'<div class="chat-bubble {role_class}">{ "You: " if msg["role"] == "user" else "Bot: " }{msg["content"]}</div>', unsafe_allow_html=True)

# Clear button - Styled
if st.button("🗑️ Clear Chat", type="secondary"):
    st.session_state.messages = []
    st.rerun()