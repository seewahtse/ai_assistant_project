"""
Step 1:
For realtime speech-to-text:
Sign up for a Free AssemblyAI API Key: 
https://www.assemblyai.com/?utm_source=youtube&utm_medium=referral&utm_campaign=yt_smit_28

For DeepSeek's R1 model:
Download Ollama: https://ollama.com

For text-to-speech:
Sign up for ElevenLabs

Install PortAudio, which is required for real-time transcription:  

    Debian/Ubuntu: apt install portaudio19-dev  
    MacOS: brew install portaudio 

Additionally, Install Python Libraries  
Before running this script, install the required dependencies:

    
 
    pip install ollama
    pip install elevenlabs 

If you're on MacOS, also install MPV for audio streaming:  

    brew install mpv

Step 2: Download the DeepSeek R1 Model  
Since this script uses DeepSeek R1 via Ollama, download the model locally by running:  

    ollama pull deepseek-r1:7b
    ollama pull llama2-uncensored

Step 3: Real-Time Transcription with AssemblyAI  
The script captures real-time audio from the microphone and converts speech to text using AssemblyAI.  
This transcription is then sent to the AI model for processing.  

Step 4: AI Response with DeepSeek R1  
Once a transcript is generated, it is sent to DeepSeek R1 (7B model) via Ollama.  
The model generates a response, which is then converted into speech using ElevenLabs.  

Step 5: Live Audio Streaming  
The generated response is streamed back to the user in real-time as speech, using ElevenLabs' text-to-speech engine.  
"""
from datetime import datetime
from errno import errorcode
from logging import exception, error

import assemblyai as aai
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import ollama
import constants as c


class AIVoiceAgent:
    def __init__(self):
        aai.settings.api_key = c.assembly_ai_key
        self.client = ElevenLabs(
            api_key=c.eleven_labs_key
        )
        self.transcriber = None
        self.log_file = "conversation_log.txt"  # Log file to store conversations

        # self.full_transcript = [
        #     {"role": "system", "content": "You are an unfiltered grumpy old sarcastic man"},
        # ]

        self.full_transcript=self.load_conversation_log()

    def load_conversation_log(self):
        messages = []
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        timestamp, rest = line.split(" - ", 1)
                        role, content = rest.split(": ", 1)
                        messages.append({"role": role.strip(), "content": content.strip()})
                    except ValueError:
                        print("improperly formatted line")
        except FileNotFoundError:
            print("Log file not found. Starting fresh.")
        return messages



    def log_conversation(self, role, content):
        with open(self.log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {role}: {content}\n")

    def start_transcription(self):
        print(f"\nReal-time transcription: ", end="\r\n")
        try:
            self.transcriber = aai.RealtimeTranscriber(
                sample_rate=44100,
                on_data=self.on_data,
                on_error=self.on_error,
                on_open=self.on_open,
                on_close=self.on_close,
                end_utterance_silence_threshold=3000

            )
            self.transcriber.connect()
            print("Connected to transcriber.")

            microphone_stream = aai.extras.MicrophoneStream(sample_rate=44100)
            self.transcriber.stream(microphone_stream)
            print("Microphone stream started.")

        except Exception as e:
            print("An error occurred while starting transcription:", str(e))

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        # print("Session ID:", session_opened.session_id)
        return

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            print(transcript.text)
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        # print("An error occured:", error)
        return

    def on_close(self):
        # print("Closing Session")
        return

    def generate_ai_response(self, transcript):
        self.stop_transcription()

        user_input = transcript.text
        self.full_transcript.append({"role": "user", "content": user_input})
        print(f"\nUser:{transcript.text}", end="\r\n")

        # Log user input
        self.log_conversation("user", user_input)


        ollama_stream = ollama.chat(
            model="llama2-uncensored",
            messages=self.full_transcript,
            stream=True,
        )

        print("llama2-uncensored:", end="\r\n")
        text_buffer = ""
        full_text = ""
        for chunk in ollama_stream:
            text_buffer += chunk['message']['content']
            if text_buffer.endswith('.'):
                audio_stream = self.client.generate(text=text_buffer,
                                                    model="eleven_turbo_v2",
                                                    stream=True)
                print(text_buffer, end="\n", flush=True)
                stream(audio_stream)
                full_text += text_buffer
                text_buffer = ""

        if text_buffer:
            audio_stream = self.client.generate(text=text_buffer,
                                                model="eleven_turbo_v2",
                                                stream=True)
            print(text_buffer, end="\n", flush=True)
            stream(audio_stream)
            full_text += text_buffer

        self.full_transcript.append({"role": "assistant", "content": full_text})

        # Log assistant response
        self.log_conversation("assistant", full_text)

        self.start_transcription()


ai_voice_agent = AIVoiceAgent()
ai_voice_agent.start_transcription()
