import sounddevice as sd
import numpy as np

# Parameters
duration = 5  # seconds
sample_rate = 44100  # Sample rate

def record_audio():
    print("Recording...")

    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()  # Wait until recording is finished

    print("Recording complete.")
    return audio_data

def main():
    audio = record_audio()

    # Optionally, play back the recorded audio
    print("Playing back recorded audio...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Wait until playback is finished

if __name__ == "__main__":
    main()