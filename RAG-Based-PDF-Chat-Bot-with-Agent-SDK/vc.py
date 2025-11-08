# # Attempting to load Whisper/gTTS 
# import whisper
# from gtts import gTTS
# import tempfile
    
# # Load Whisper only once on startup
# print("Loading Whisper 'base' model...")
# # This might take a moment, but it's done once globally.
# model = whisper.load_model("base")
# print("Whisper model loaded.")
    
# def speech_to_text(audio_path: str) -> str:
#     """Converts audio to text using Whisper (offline + free)."""
#     print(f"STT: Transcribing {audio_path}...")
#     try:
#         # Transcribe the audio file
#         result = model.transcribe(audio_path)
#         # The result is a dictionary, extract the text
#         return result["text"]
#     except Exception as e:
#         print(f"Error during Whisper transcription: {e}")
#         # In case of transcription error, return a clear error message
#         return f"Error transcribing audio: {e}"
    
# def clean_for_voice_output(raw_text: str) -> str:
#     """
#     Removes prefixes, citations, and other formatting from RAG or agent outputs
#     before TTS.
#     """
#     if '\n' in raw_text:
#         spoken_text = raw_text[raw_text.find('\n'):].strip()
#     else:
#         spoken_text = raw_text

#     citation_start = spoken_text.find('[DOCUMENT SOURCE CITATIONS:')
#     if citation_start != -1:
#         spoken_text = spoken_text[:citation_start].strip()

#     return spoken_text or raw_text


# def text_to_speech(text: str) -> str:
#     """
#     Converts text to voice using gTTS (free) and returns a temporary MP3 path.
#     The temporary file uses delete=False so FastAPI can serve it, requiring
#     manual cleanup later (handled in the voice chat endpoint logic).
#         """
#     print("TTS: Generating audio with gTTS...")
#     try:
#         # Initialize gTTS object
#         tts = gTTS(text=text, lang="en")
#         # Create a temporary file path
#         # We must close the file handle before gTTS saves to the path.
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#         temp_path = temp_file.name
#         temp_file.close()
            
#         # Save the TTS audio to the temporary path
#         tts.save(temp_path)
            
#         return temp_path
#     except Exception as e:
#         print(f"Error during gTTS generation: {e}")
#         return "" # Return empty string on failure
        


# print(f"VOICE DEPENDENCY ERROR: {e}. Voice features will be disabled.")
# # Fallback to dummy functions if imports fail
# def speech_to_text(audio_path: str) -> str:
#     print("STT FAILED: Voice dependencies missing.")
#     return "I am unable to process audio at the moment."

# def text_to_speech(text: str) -> str:
#     print("TTS FAILED: Voice dependencies missing.")
#     return "" # No audio path to return