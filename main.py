from faster_whisper import WhisperModel
import openai, wave, os, torch
from TTS.api import TTS
import sounddevice as sd
import simpleaudio as sa
import numpy as np
import time, keyboard, pyaudio


# Constants
output_filename = "recorded_audio.wav"
DEBUG = False

# Get device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Faster-Whisper
model_size = "medium.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Init TTS
print("start")
model_name = TTS().list_models()
for index, item in enumerate(model_name, 0):
    print(f"{index}. {item}")
model_number = 10
model_name = TTS().list_models()[model_number]
my_list = [17, 11, 10, 14, 15]
if model_number in my_list:
    device='cpu'
tts = TTS(model_name).to(device)

# OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
chat_history = []
initial_prompt = """You are an AI assistant of English learning.  
Use beginner-level vocabulary. 
You make only one question and wait for my answer. 
Teach me english learning new words and sentences. 
No enumerate the questions. 
You have the power of your creativity for new kinds for learn English. 
Remember that I have 5 years old. """
initial_prompt = initial_prompt.replace("\n", "")
STATE = "AI"
PROMPT = ""
collected_messages = []
full_reply_content_last = ""


# Constants for audio recording
fs = 22050  # Sample rate (samples per second)
duration = 60  # Maximum recording duration in seconds
recording = False
audio_data = []
AUDIO_RECORDED=False
stream = ""

# Callback function to record audio
def audio_callback(in_data, frame_count, time_info, status):
    global audio_data
    if status:
        print(status)
    if recording:
        audio_data.append(in_data)
    return in_data, pyaudio.paContinue

# Function to start or stop recording
def toggle_recording(self):
    global recording
    global stream
    p = pyaudio.PyAudio()
    global audio_data
    if not recording:
        print("Recording started...")
        audio_data=[]
        recording = True
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=fs,
                        input=True,
                        frames_per_buffer=1024,
                        stream_callback=audio_callback)
        stream.start_stream()
    else:
        print("Recording stopped.")
        recording = False
        stream.stop_stream()
        stream.close()
        p.terminate()
        save_audio_to_wav()

# Function to save recorded audio to a WAV file
def save_audio_to_wav():
    global audio_data
    global AUDIO_RECORDED
    if audio_data:
        print(f"Saving recorded audio to '{output_filename}'...")
        audio_data = b''.join(audio_data)
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio_data)
        print(f"Recording saved as '{output_filename}'.")
        AUDIO_RECORDED = True


def chatGPT(prompt):
    global STATE
    global full_reply_content_last
    collected_messages =[]
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
        if DEBUG:
            print(full_reply_content)
        # clear the terminal
        print("\033[H\033[J", end="")
    chat_history.append({"role": "assistant", "content": full_reply_content})
    # print the time delay and text received
    full_reply_content_last = full_reply_content
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    if DEBUG:
        print(f"GPT: {full_reply_content}")
    full_reply_content = ''
    saveChatHistory()
    STATE="TTS"

def saveChatHistory():
    global chat_history
    # Specify the file path
    file_path = "output.txt"
    # Open the file in write mode
    with open(file_path, "w") as file:
    # Use a loop to write each element to the file
        file.write(str(chat_history))  # Add a newline after each element

def textoSpeach(text_to):
    global STATE 
    #tts.tts_to_file(text=text_to,  speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")
    tts.tts_to_file(text=text_to,   file_path="output.wav")
    wave_obj = sa.WaveObject.from_wave_file("output.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()
    #print("\033[H\033[J", end="")
    #print("Listen to me")
    time.sleep(2.2)
    play_obj = wave_obj.play()
    play_obj.wait_done()
    STATE="USER"


def Whisper():
    global PROMPT
    print("enter whisper")
    segments, info = model.transcribe("recorded_audio.wav", beam_size=5)
    result = ""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        result = result + segment.text
    print(result)
    PROMPT = result
    time.sleep(1.05)
    
def Sumarize():
    file_path = "output.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            file_contents = file.read() # Add a newline after each element
            file_path = "chat_history.txt"
            # Open the file in write mode
            with open(file_path, "a") as file:
                # Use a loop to write each element to the file
                file.write(file_contents)  # Add a newline after each element
            
            prompt = "sumarize this text without number:" + file_contents
            chat_history.append({"role": "user", "content": prompt})
            response_iterator = openai.ChatCompletion.create(
                model="gpt-4",
                messages = chat_history,
                stream=True,
                max_tokens=300)
            for chunk in response_iterator:
                chunk_message = chunk['choices'][0]['delta']  # extract the message
                collected_messages.append(chunk_message)  # save the message
                full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
            print("This is a sumarize of last classrom: " + full_reply_content + " and this is your task: " + initial_prompt)
            time.sleep(0.1)
            chatGPT("This is a sumarize of last classroom: " + full_reply_content + ". And this is your task: " + initial_prompt)
            chat_history.pop(0)
    else:
        chatGPT(initial_prompt)



#Sumarize()

chatGPT(initial_prompt)

while True:

    if STATE == "AI":
        chatGPT(PROMPT)

    if STATE == "USER":
        AUDIO_RECORDED = False
        keyboard.on_press_key('space', toggle_recording)
        while AUDIO_RECORDED==False:
            time.sleep(0.05)
        keyboard.unhook_all()
        STATE="WHISPER"

    if STATE == "TTS":
        textoSpeach(full_reply_content_last)

    if STATE == "WHISPER":
        Whisper()
        STATE="AI"

    time.sleep(0.05)