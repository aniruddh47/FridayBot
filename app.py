import streamlit as st
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time
from groq import Groq
import tempfile
from pathlib import Path
import base64

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Debug logging
print(f"API Key loaded: {'Yes' if groq_api_key else 'No'}")
if groq_api_key:
    print(f"API Key length: {len(groq_api_key)}")
    print(f"API Key starts with: {groq_api_key[:10]}...")

# List of supported models (Updated)
SUPPORTED_MODELS = [
    "llama-3.1-8b-instant",  # Fast Llama model
    "deepseek-r1-distill-qwen-32b"  # DeepSeek model
]

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.mp4', '.ogg', '.flac']

# Available voices for text-to-speech
AVAILABLE_VOICES = [
    "Arista-PlayAI",
    
]

# Model descriptions for better user understanding
MODEL_DESCRIPTIONS = {
    "llama-3.1-8b-instant": "Fast, efficient model for quick responses",
    "deepseek-r1-distill-qwen-32b": "Advanced distilled model with excellent performance"
}

def transcribe_audio(audio_file):
    """Transcribe audio file using Groq's Whisper model"""
    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name

        # Transcribe the audio file
        with open(tmp_file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(tmp_file_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        return transcription.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def text_to_speech(text, voice):
    """Convert text to speech using Groq's TTS model"""
    try:
        # Generate speech from text
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        # The response is a BinaryAPIResponse object, we need to read it
        return response.read()
        
    except Exception as e:
        return f"Error generating speech: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'model' not in st.session_state:
        st.session_state.model = SUPPORTED_MODELS[0]
    if 'memory_length' not in st.session_state:
        st.session_state.memory_length = 15
    if 'active_mode' not in st.session_state:
        st.session_state.active_mode = "chatbot"  # Default mode
    if 'selected_voice' not in st.session_state:
        st.session_state.selected_voice = AVAILABLE_VOICES[0]

def create_conversation(model, memory_length):
    """Create a new conversation with the specified model and memory"""
    memory = ConversationBufferWindowMemory(k=memory_length)
    
    # Preload existing chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})
    
    # Initialize Groq chat model
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )
    
    return ConversationChain(
        llm=groq_chat,
        memory=memory
    )

def handle_user_input(user_question):
    """Process user input and generate response"""
    if not user_question.strip():
        return
    
    # Display user message
    st.chat_message("user").write(user_question)
    
    # Display thinking indicator
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.text("Thinking...")
        
        try:
            # Get response from model
            response = st.session_state.conversation.invoke({'input': user_question})
            chatbot_reply = response['response']
            
            # Save to chat history
            message = {'human': user_question, 'AI': chatbot_reply}
            st.session_state.chat_history.append(message)
            
            # Replace thinking indicator with actual response
            thinking_placeholder.empty()
            st.write(chatbot_reply)
            
        except Exception as e:
            thinking_placeholder.empty()
            st.error(f"⚠️ Error: {str(e)}")

def reset_conversation():
    """Clear conversation history and reset the chat"""
    st.session_state.chat_history = []
    st.session_state.conversation = create_conversation(
        st.session_state.model, 
        st.session_state.memory_length
    )

def handle_model_change():
    """Handle model change and recreate conversation"""
    st.session_state.conversation = create_conversation(
        st.session_state.model, 
        st.session_state.memory_length
    )

def main():
    # Page configuration
    st.set_page_config(
        page_title="Friday",
        page_icon="👾",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    with st.sidebar:
        st.title('MODES')
        
        # Mode selection in sidebar
        st.subheader('🔄 Mode Selection')
        st.markdown("Choose the active mode:")
        
        # Chatbot button
        if st.button("💬 Chatbot Mode", key="chatbot_btn", use_container_width=True):
            st.session_state.active_mode = "chatbot"
            st.rerun()
        
        # Speech to Text button
        if st.button("🎤 Speech to Text Mode", key="transcription_btn", use_container_width=True):
            st.session_state.active_mode = "speech_to_text"
            st.rerun()
        
        # Text to Speech button
        if st.button("🔊 Text to Speech Mode", key="tts_btn", use_container_width=True):
            st.session_state.active_mode = "text_to_speech"
            st.rerun()
        
        # Display current mode
        st.markdown(f"**Current Mode**: {st.session_state.active_mode.replace('_', ' ').title()}")
        
        # Memory settings
        st.subheader('')
        memory_length = 5
        # Update memory length if changed
        if memory_length != st.session_state.memory_length:
            st.session_state.memory_length = memory_length
            handle_model_change()
        
        # Reset button
        st.button("New Conversation", on_click=reset_conversation)
        
        # API key status indicator
       
    
    # Main interface
    def set_bg(image_path):
        """Convert image to Base64 and set it as background."""
        with open(image_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode()
            
            bg_style = f"""
            <style>
            /* Base styles */
            :root {{
                --primary-color: #00d4ff;
                --secondary-color: #090979;
                --bg-color: #0a0a0a;
                --glass-bg: rgba(255, 255, 255, 0.1);
                --text-color: #ffffff;
            }}
            
            [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_img}");
            background-size: cover;
            background-position: right bottom;
            background-repeat: no-repeat;
            background-attachment: scroll;
            min-height: 100vh;
            overflow-y: auto;
            }}
            
            [data-testid="stAppViewContainer"]::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, rgba(10, 10, 10, 0.85), rgba(26, 26, 26, 0.85));
                z-index: 0;
                pointer-events: none;
            }}
            
            /* Ensure content is above the overlay */
            .main .block-container,
            .stChatMessage,
            .stTextInput,
            .stButton,
            .css-1d391kg,
            .css-1v0mbdj {{
                position: relative;
                z-index: 1;
            }}
            
            /* Chat messages styling */
            .stChatMessage {{
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: transform 0.3s ease;
            }}
            
            .stChatMessage:hover {{
                transform: translateY(-2px);
            }}
            
            /* User message specific styling */
            .stChatMessage[data-testid="user"] {{
                background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
                margin-left: 20%;
            }}
            
            /* Bot message specific styling */
            .stChatMessage[data-testid="assistant"] {{
                background: var(--glass-bg);
                margin-right: 20%;
            }}
            
            /* Input field styling */
            .stTextInput > div > div > input {{
                background: var(--glass-bg) !important;
                backdrop-filter: blur(10px);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: var(--text-color) !important;
                padding: 12px;
                transition: all 0.3s ease;
            }}
            
            .stTextInput > div > div > input:focus {{
                box-shadow: 0 0 15px var(--primary-color);
                border-color: var(--primary-color);
            }}
            
            /* Button styling */
            .stButton > button {{
                background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
                color: var(--text-color) !important;
                border-radius: 15px;
                border: none;
                padding: 10px 20px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
            }}
            
            /* Sidebar styling */
            .css-1d391kg, .css-1v0mbdj {{
                background: var(--glass-bg) !important;
                backdrop-filter: blur(10px);
                border-right: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            /* Sidebar elements */
            .css-1d391kg .stMarkdown, .css-1v0mbdj .stMarkdown {{
                color: var(--text-color) !important;
                font-weight: 500;
            }}
            
            /* Main content area */
            .main .block-container {{
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                padding: 2rem;
                border-radius: 20px;
                margin: 1rem;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            /* Typography */
            .stMarkdown, .stText, .stTitle, h1, h2, h3, h4, h5, h6 {{
                color: var(--text-color) !important;
                font-family: 'Inter', sans-serif;
            }}
            
            /* Mode selection buttons */
            .stButton > button[data-testid="baseButton-secondary"] {{
                background: var(--glass-bg) !important;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 5px 0;
                width: 100%;
            }}
            
            .stButton > button[data-testid="baseButton-secondary"]:hover {{
                background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
            }}
            
            /* Animated dots */
            .dots {{
                display: inline-block;
                color: var(--primary-color);
            }}
            
            .dots:after {{
                content: '';
                animation: dots 1.5s steps(4, end) infinite;
            }}
            
            @keyframes dots {{
                0% {{ content: ''; }}
                25% {{ content: '.'; }}
                50% {{ content: '..'; }}
                75% {{ content: '...'; }}
                100% {{ content: '...'; }}
            }}
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {{
                .stChatMessage[data-testid="user"] {{
                    margin-left: 10%;
                }}
                .stChatMessage[data-testid="assistant"] {{
                    margin-right: 10%;
                }}
                .main .block-container {{
                    margin: 0.5rem;
                    padding: 1rem;
                }}
                [data-testid="stAppViewContainer"] {{
                    background-size: 70%;
                }}
            }}
            </style>
            """
            st.markdown(bg_style, unsafe_allow_html=True)

    # Set background
    set_bg("friday.png")
    st.markdown("""
        <h1 style="text-align: center; font-size: 2.5em; margin-bottom: 1em;">
            It's Friday<span class='dots'></span>
        </h1>
    """, unsafe_allow_html=True)
    
    # Display active mode
    st.markdown(f"**Current Mode**: {st.session_state.active_mode.replace('_', ' ').title()}")
    
    # Initialize conversation if not already done
    if st.session_state.conversation is None:
        st.session_state.conversation = create_conversation(
            st.session_state.model, 
            st.session_state.memory_length
        )
    
    # Check API key
    if not groq_api_key:
        st.error("API key is missing. Please check your .env file.")
        return
    
    # Display content based on active mode
    if st.session_state.active_mode == "chatbot":
        # Display chat history
        for message in st.session_state.chat_history:
            st.chat_message("user").write(message['human'])
            st.chat_message("assistant").write(message['AI'])
        
        # Chat input
        user_question = st.chat_input("Ask a question...")
        if user_question:
            handle_user_input(user_question)
    
    elif st.session_state.active_mode == "speech_to_text":
        # Speech to Text: Only file upload
        st.subheader("📂 Upload Audio File for Transcription")

        st.write("Upload an audio file for transcription:")
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=[fmt[1:] for fmt in SUPPORTED_AUDIO_FORMATS],
            help="Supported formats: MP3, WAV, M4A, MP4, OGG, FLAC"
        )

        if audio_file:
            with st.spinner("📝 Transcribing audio..."):
                transcription = transcribe_audio(audio_file)
                st.subheader("📜 Transcription:")
                st.text_area("Transcribed Text", transcription, height=150, key="uploaded_transcription")

            # Text-to-Speech playback
            if st.button("🔊 Convert to Speech", key="tts_uploaded"):
                with st.spinner("🎙️ Generating Speech..."):
                    response = groq_client.audio.speech.create(
                        model="playai-tts",
                        voice=st.session_state.selected_voice,
                        input=transcription,
                        response_format="wav"
                    )
                    st.audio(response.read(), format="audio/wav")
                    st.success("✅ Speech Generated!")

            # Add transcription to chat
            if st.button("💬 Add to Chat", key="add_uploaded_to_chat"):
                st.session_state.active_mode = "chatbot"
                handle_user_input(f"Transcribed text: {transcription}")

    
    elif st.session_state.active_mode == "text_to_speech":
        # Text to Speech section
        st.subheader("🔊 Text to Speech")
        
        # Voice selection
        st.session_state.selected_voice = st.selectbox(
            "Text to voice",
            AVAILABLE_VOICES,
            index=AVAILABLE_VOICES.index(st.session_state.selected_voice)
        )
        
        # Text input
        text_input = st.text_area(
            "Enter text to convert to speech",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        # Generate speech button
        if st.button("Generate Speech", disabled=not text_input.strip()):
            with st.spinner("Generating speech..."):
                audio_bytes = text_to_speech(text_input, st.session_state.selected_voice)
                
                if isinstance(audio_bytes, str) and audio_bytes.startswith("Error"):
                    st.error(audio_bytes)
                else:
                    # Create a download button for the audio
                    b64_audio = base64.b64encode(audio_bytes).decode()
                    href = f'<a href="data:audio/wav;base64,{b64_audio}" download="generated_speech.wav">Download Audio</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Play the audio
                    st.audio(audio_bytes, format="audio/wav")

if __name__ == "__main__":
    main()