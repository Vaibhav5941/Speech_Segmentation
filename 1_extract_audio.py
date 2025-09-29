import os
import subprocess
from pathlib import Path
from scipy.io import wavfile
import numpy as np

def extract_audio(input_file, output_dir="output"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_file = os.path.join(output_dir, "extracted_audio.wav")
    file_ext = Path(input_file).suffix.lower()

    print("Step 1: Extracting audio...")

    if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']:
        cmd = ['ffmpeg', '-i', input_file, '-vn', '-acodec', 'pcm_s16le',
               '-ar', '16000', '-ac', '1', '-y', audio_file]
    elif file_ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
        cmd = ['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le',
               '-ar', '16000', '-ac', '1', '-y', audio_file]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    subprocess.run(cmd, check=True)
    print(f"✓ Audio saved to: {audio_file}")

    sample_rate, audio_data = wavfile.read(audio_file)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
    return audio_file

if __name__ == "__main__":
    input_file = "test_audio.wav"  # change if needed
    extract_audio(input_file)
