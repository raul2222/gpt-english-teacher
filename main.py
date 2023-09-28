import openai
import keyboard
import pyaudio
import wave
import os
import torch
from TTS.api import TTS
import simpleaudio as sa
import time

# Get device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

model_name = TTS().list_models()
for index, item in enumerate(model_name, 0):
    print(f"{index}. {item}")
model_name = TTS().list_models()[8]
# Init TTS
tts = TTS(model_name).to(device)


# OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
chat_history = []
initial_prompt = """you are an hiring manager. 
you'll use beginner-level vocabulary. In this role,
asking you questions one at a time and waiting for your responses. 
The job we're discussing is in the technology field."""
STATE = "AI"
PROMPT = ""
collected_messages = []
full_reply_content = ""

# Constants
output_filename = "captured_audio.wav"
trigger_key = "p"  # Change this to the key you want to use as a trigger

# Audio settings
sample_rate = 22050  # Lower the sample rate
chunk_size = 4096   # Increase the chunk size
audio_format = pyaudio.paInt16
audio = pyaudio.PyAudio() # Initialize PyAudio

'''
def captureAudio():
    stream = audio.open(format=audio_format,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    # Wait for the trigger key press
    keyboard.wait(trigger_key)
    frames = []
    while True:
        try:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
        except KeyboardInterrupt:
            break
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    stream.stop_stream()
    stream.close()
    audio.terminate()
    STATE="WHISPER"
'''

def chatGPT(prompt):
    global STATE
    global full_reply_content
    global collected_messages
    chat_history.append({"role": "user", "content": prompt})
    response_iterator = openai.ChatCompletion.create(
        model="gpt-4",
        messages = chat_history,
        stream=True,
        max_tokens=300,
    )
    
    for chunk in response_iterator:
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        print(full_reply_content)
        # clear the terminal
        print("\033[H\033[J", end="")
    chat_history.append({"role": "assistant", "content": full_reply_content})
    # print the time delay and text received
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    print(f"GPT: {full_reply_content}")
    STATE="TTS"
    print("exit gpt")


def textoSpeach(text_to):
    global STATE 
    global full_reply_content
    print(text_to)
    tts.tts_to_file(text=text_to, file_path="output.wav")
    wave_obj = sa.WaveObject.from_wave_file("output.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()
    STATE="USER"

'''
def Whisper():
    STATE="AI"
'''
chatGPT(initial_prompt)

while True:
    if STATE == "AI":
        chatGPT(PROMPT)
    #if STATE == "USER":
    #    captureAudio()
    if STATE == "TTS":
        textoSpeach(full_reply_content)
    #if STATE == "WHISPER":
    #    Whisper()