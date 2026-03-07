import os
from dotenv import load_dotenv
from google import genai
import sounddevice as sd
import io
from google.genai import types
from scipy.io.wavfile import write
import pyttsx3

fs = 16000
seconds = 3

print("Recording...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

# ✅ Play it back immediately
print("Playing back...")
sd.play(audio, samplerate=fs)
sd.wait()

# Convert audio to bytes in memory
buffer = io.BytesIO()
write(buffer, fs, audio)

audio_bytes = buffer.getvalue()
# audio_base64 = base64.b64encode(audio_bytes).decode()

print("Audio ready to send to LLM")

if "SSLKEYLOGFILE" in os.environ:
    del os.environ["SSLKEYLOGFILE"]

# load variables from .env
load_dotenv()

# get api key
api_key = os.getenv("GEMINI_API_KEY")



# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=api_key)

# print("List of models that support generateContent:\n")
# for m in client.models.list():
#     for action in m.supported_actions:
#         if action == "generateContent":
#             print(m.name)

response = client.models.generate_content(
    model="models/gemini-3.1-flash-lite-preview",      
    contents=[
        types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        )
    ],
)



# Assuming 'response' holds the text returned from the LLM
response_text = response.text

# Initialize the TTS engine
engine = pyttsx3.init()

# Optionally set properties like rate or voice
engine.setProperty('rate', 150)  # Speed of speech

# Convert text to speech
engine.say(response_text)
engine.runAndWait()

# print("response : ",response.model_dump_json(indent=2))
print("response:", response.text)

print("---------")
