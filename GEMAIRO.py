import requests
import tempfile
import time
import os
import audioread
import pyaudio
import pygame
import sys
import numpy as np
import sounddevice as sd
import threading
import speech_recognition
import pyttsx3
import replicate

# Set your API key
client = replicate.Client(api_token=os.environ.get("REPLICATE_API_KEY"))

elevenLabsKey = os.environ.get("xi_key")
# Initialize Pygame
pygame.init()

# Set up display
window = pygame.display.set_mode((500, 500))
width, height = window.get_size()
pygame.display.set_caption("Gemairo")

# Set up colors
BLACK = (0, 0, 0)
GREEN = (0, 155, 0)

# Initialize global variables
mouthHeight = 50  # Initial mouth height
mouth_lock = threading.Lock()  # Lock for thread-safe updates

# Constants
CHUNK_SIZE = 1024

# ElevenLabs API settings
url = "https://api.elevenlabs.io/v1/text-to-speech/rmgiMEglLwlWGqi9FkKW"
headers = {
    "Accept": "audio/wav",
    "Content-Type": "application/json",
    "xi-api-key": elevenLabsKey
}

recognizer = speech_recognition.Recognizer()

# Track the last processed message
lastTime = None

# Set the temp directory for debug files
tempfile.tempdir = os.path.expanduser('~/Documents/Programming/GEMAIRO/debug')

def retrieve_first_message():
    # Ensure the output folder exists
    output_folder = tempfile.gettempdir()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                print("Listening...")
                audio = recognizer.listen(mic)

                text = recognizer.recognize_google(audio)
                text = text.lower()
                print(f"Recognized: {text}")
                input_data = {
                    "top_p": 0.9,
                    "prompt": text,
                    "min_tokens": 0,
                    "temperature": 0.6,
                    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are Gemairo, you are a chatbot Built by your creator: Joris, you are a bro, a homie and a best friend, you are not corny at all and your messages are quite direct, but sometimes you get really unhinged, you are not allowed to use '*' at all, just talk with words, no roleplay at all<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "presence_penalty": 1.15
                }

                # Initialize a variable to accumulate the text
                full_response = ""

                # Stream the response from the model
                for event in client.stream("meta/meta-llama-3-70b-instruct", input=input_data):
                    # Accumulate the text
                    full_response += event.data

                # After the loop, send the accumulated text to ElevenLabs
                data = {"text": full_response}
                response = requests.post(url, json=data, headers=headers)
                
                # Define the path to save the output file
                output_filename = os.path.join(output_folder, 'message.wav')

                with open(output_filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            file.write(chunk)

                play_audio(output_filename)
                os.remove(output_filename)

        except speech_recognition.UnknownValueError:
            print("Could not understand the audio, please try again.")
            continue
        except Exception as e:
            print(f"An error occurred: {e}")
            break


def play_audio(filename):
    try:
        with audioread.audio_open(filename) as f:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                            channels=f.channels,
                            rate=f.samplerate,
                            output=True)

            # Signal that audio playback is starting
            monitor.set_playing_audio(True)

            for buf in f:
                stream.write(buf)

            # Signal that audio playback is stopping
            monitor.set_playing_audio(False)

            stream.stop_stream()
            stream.close()
            p.terminate()

    except audioread.DecodeError:
        print(f"Error: Unable to decode {filename}. The file may be corrupted or unsupported.")

class SoundMonitor:
    def __init__(self):
        self.noise = 0
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.monitoring = False
        self.playing_audio = False
        self.playback_lock = threading.Lock()
        print("SoundMonitor initialized.")

    def monitor_sound(self):
        print("Starting sound monitoring. Press Ctrl+C to stop.")
        self.monitoring = True

        try:
            with sd.InputStream(callback=self.audio_callback, samplerate=self.sample_rate,
                                blocksize=self.chunk_size, channels=1, dtype='float32') as stream:
                print("InputStream started.")
                while self.monitoring:
                    if self.playing_audio:
                        # Only monitor when audio is being played
                        sd.sleep(1000)
                    else:
                        # Sleep longer if not playing audio to reduce CPU usage
                        sd.sleep(5000)
                    print("Monitoring...")

        except KeyboardInterrupt:
            print("Monitoring stopped by user.")
            self.stop_recording()

        except Exception as e:
            print(f"Error occurred: {e}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")

        if self.monitoring and self.playing_audio:
            # Calculate the root mean square (RMS) of the audio chunk for better sensitivity
            rms = np.sqrt(np.mean(np.square(indata)))
            max_val = rms * 5000  # Increase the sensitivity by scaling factor

            self.noise = max_val

            # Update the mouth height based on noise level, within the lock
            with mouth_lock:
                global mouthHeight
                mouthHeight = max(10, min(300, int(max_val)))  # Cap mouthHeight for drawing

    def set_playing_audio(self, status):
        with self.playback_lock:
            self.playing_audio = status

    def stop_recording(self):
        self.monitoring = False
        print("Sound monitoring stopped.")

def draw_face():
    global mouthHeight

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        window.fill(BLACK)

        pygame.draw.rect(window, GREEN, (420, 140, 50, 50))
        pygame.draw.rect(window, GREEN, (210, 140, 50, 50))

        mouth_width = 200
        mouth_x = (window.get_width() - mouth_width) // 2
        mouth_y = 300

        with mouth_lock:
            pygame.draw.rect(window, GREEN, (mouth_x, mouth_y, mouth_width, mouthHeight))

        pygame.display.update()
        pygame.time.Clock().tick(60)

def main():
    global monitor  # Make monitor global to access in play_audio function

    monitor = SoundMonitor()

    message_thread = threading.Thread(target=retrieve_first_message)
    message_thread.daemon = True
    message_thread.start()

    sound_thread = threading.Thread(target=monitor.monitor_sound)
    sound_thread.daemon = True
    sound_thread.start()

    draw_face()

if __name__ == '__main__':
    main()

