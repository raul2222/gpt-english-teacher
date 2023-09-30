import openai
import wave
import os
import torch
from TTS.api import TTS
import simpleaudio as sa
import time
import keyboard


# Get device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# Init TTS
print("start")
model_name = TTS().list_models()
for index, item in enumerate(model_name, 0):
    print(f"{index}. {item}")
model_number = 13
model_name = TTS().list_models()[model_number]
my_list = [17, 11, 10, 14, 15]
if model_number in my_list:
    device='cpu'
tts = TTS(model_name).to(device)


# OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
chat_history = []
# you'll use medium-level vocabulary. In this role,
initial_prompt = """you name is Eva and you are an AI of English learning. 
use beginner-level vocabulary.
you make only one question and wait for my answer. 
We have 5 years old.
No enumerate the questions."""
STATE = "AI"
PROMPT = ""
collected_messages = []
full_reply_content = ""

# Constants
output_filename = "captured_audio.wav"
trigger_key = "p"  # Change this to the key you want to use as a trigger

# Constants for audio recording
fs = 44100  # Sample rate (samples per second)
duration = 60  # Maximum recording duration in seconds
output_filename = "recorded_audio.wav"
recording = False
audio_data = []


# Function to start or stop recording
def toggle_recording(self):
    global recording
    global audio_data

    if not recording:
        print("Recording started...")
        audio_data = []
        recording = True
    else:
        recording = False
        save_audio_to_wav()
        print("Recording stopped.")
        keyboard.unhook_all()

# Function to save recorded audio to a WAV file
def save_audio_to_wav():
    if audio_data:
        print(f"Saving recorded audio to '{output_filename}'...")
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(b''.join(audio_data))
        print(f"Recording saved as '{output_filename}'.")

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


def textoSpeach(text_to):
    global STATE 
    global full_reply_content
    #tts.tts_to_file(text=text_to,  speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")
    tts.tts_to_file(text=text_to,   file_path="output.wav")
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
    if STATE == "USER":
        keyboard.on_press_key('r', toggle_recording)
        keyboard.wait('q')
        STATE="WHISPER"
    if STATE == "TTS":
        textoSpeach(full_reply_content)
    if STATE == "WHISPER":
        #Whisper()
        print("wisperq")
    time.sleep(0.05)