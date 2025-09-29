import os
import json
from pathlib import Path
from scipy.io import wavfile
import numpy as np

def export_segments(audio_file, json_file, output_dir="output/segments"):
    print("\nStep 3: Exporting audio segments...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_rate, audio_data = wavfile.read(audio_file)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    with open(json_file, "r") as f:
        segments = json.load(f)

    for idx, seg in enumerate(segments, 1):
        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)
        segment_data = audio_data[start_sample:end_sample]
        segment_data_int = (segment_data * 32768.0).astype(np.int16)

        segment_filename = f"segment_{idx:02d}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        wavfile.write(segment_path, sample_rate, segment_data_int)

        print(f"  ✓ {segment_filename} ({seg['end'] - seg['start']:.2f}s)")

    print(f"\n✓ All segments exported to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    audio_file = "output/extracted_audio.wav"
    json_file = "output/speech_segments.json"
    export_segments(audio_file, json_file)
