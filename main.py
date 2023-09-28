import openai
import keyboard
import pyaudio
import wave
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

chat_history = []

initial_prompt = """you are an employer with a focus on job interview scenarios. 
For our conversation, I'll use beginner-level vocabulary. In this role, I'll act as the hiring manager, asking you questions one at a time and waiting for your responses. 
The job we're discussing is in the technology field."""

# Constants
output_filename = "captured_audio.wav"
trigger_key = "p"  # Change this to the key you want to use as a trigger

# Audio settings
sample_rate = 22050  # Lower the sample rate
chunk_size = 4096   # Increase the chunk size
audio_format = pyaudio.paInt16

# Initialize PyAudio
audio = pyaudio.PyAudio()

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



def chatGPT(prompt):

    chat_history.append({"role": "user", "content": prompt})
    response_iterator = openai.ChatCompletion.create(
        model="gpt-4",
        messages = chat_history,
        stream=True,
        max_tokens=100,
    )
    collected_messages = []
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


chatGPT(initial_prompt)