import speech_recognition as sr
from google.cloud import speech
from faster_whisper import WhisperModel
import os

# --- 1. Google Cloud Speech-to-Text (GCS File) ---

def transcribe_gcs_example(gcs_uri: str):
    """Transcribes an audio file located in Google Cloud Storage."""
    print("--- 1. Google Cloud Speech-to-Text (GCS File) ---")
    
    # NOTE: This requires authentication (e.g., Application Default Credentials) 
    # and a billing-enabled project.
    
    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        print(f"Sending request for GCS file: {gcs_uri}...")
        response = client.recognize(config=config, audio=audio)

        transcript = "".join(
            result.alternatives[0].transcript for result in response.results
        )
        print(f"✅ GCS Transcription: {transcript}\n")
    except Exception as e:
        print(f"❌ GCS Example Error (Skipped): Ensure the 'google-cloud-speech' library is installed, "
              f"the API is enabled, and authentication is set up. Details: {e}\n")


# --- 2. Faster-Whisper (Local File) ---

def transcribe_whisper_example(local_file_path: str):
    """Transcribes a local audio file using the faster-whisper model."""
    print("--- 2. Faster-Whisper (Local File) ---")

    if not os.path.exists(local_file_path):
        print(f"❌ Whisper Example Error (Skipped): Local file not found at '{local_file_path}'. "
              "Please provide a valid audio file (e.g., .mp3, .wav) for this example.\n")
        return

    try:
        # Using a smaller model for a faster demo
        model_size = "base"
        # Set device="cuda" if you have a compatible NVIDIA GPU
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        print(f"Loading '{model_size}' model and transcribing local file: {local_file_path}...")
        segments, info = model.transcribe(local_file_path, beam_size=5)

        full_transcript = "".join(segment.text for segment in segments)
        print(f"✅ Whisper Transcription: {full_transcript.strip()}\n")
    except Exception as e:
        print(f"❌ Whisper Example Error (Skipped): Ensure 'faster-whisper' is installed and 'ffmpeg' is "
              f"accessible on your system PATH. Details: {e}\n")


# --- 3. SpeechRecognition (Live Microphone Input) ---

def recognize_from_mic_example():
    """Captures audio from the microphone and uses the Google Web Speech API."""
    print("--- 3. SpeechRecognition (Live Microphone Input) ---")
    
    r = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("🔊 Speak now! I'm listening... (Will stop after 5 seconds of speaking or upon detecting silence)")
            
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source)
            
            # Listen for up to 5 seconds
            audio = r.listen(source, phrase_time_limit=5)
            print("...Done listening. Recognizing speech...")
            
        # Use Google Web Speech API (uses a free, default API key for testing)
        text = r.recognize_google(audio)
        print(f"✅ Microphone Transcription: {text}")
        
    except sr.WaitTimeoutError:
        print("⏸️ Microphone Example: No speech detected within the time limit. Try again.")
    except sr.UnknownValueError:
        print("❌ Microphone Example: Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"❌ Microphone Example: Could not request results from Google Web Speech API; {e}")
    except Exception as e:
        print(f"❌ Microphone Example Error (Skipped): Ensure 'PyAudio' and 'SpeechRecognition' "
              f"are installed correctly. Details: {e}")
    print("\n" + "-"*50 + "\n")


# --- Main Execution ---

if __name__ == "__main__":
    
    print("\n" + "="*50)
    print("STARTING PYTHON SPEECH-TO-TEXT DEMO")
    print("="*50 + "\n")
    
    # 1. Configuration for GCS Example
    # This is a public audio file from Google's samples.
    GCS_URI = "gs://cloud-samples-data/speech/brooklyn_bridge.flac"
    transcribe_gcs_example(GCS_URI)

    # 2. Configuration for Whisper Example
    # NOTE: You must replace 'path/to/your/audio.mp3' with a real local audio file.
    # If the file isn't found, the script will skip this step.
    LOCAL_AUDIO_FILE = "path/to/your/audio.mp3" 
    # Example for testing if you have a WAV file named 'test_audio.wav' in the same directory:
    # LOCAL_AUDIO_FILE = "test_audio.wav"
    transcribe_whisper_example(LOCAL_AUDIO_FILE)
    # 3. Configuration for Microphone Example
    recognize_from_mic_example()
    
    print("="*50)
    print("DEMO COMPLETE")
    print("="*50)