import os
import json
import numpy as np
from scipy.io import wavfile

def detect_speech_segments(audio_file, output_dir="output",
                           frame_duration=0.03, energy_threshold=0.02,
                           min_silence_duration=0.3, min_speech_duration=0.5):
    print("\nStep 2: Detecting speech segments...")

    sample_rate, audio_data = wavfile.read(audio_file)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    frame_length = int(frame_duration * sample_rate)
    num_frames = len(audio_data) // frame_length

    energies = [np.sqrt(np.mean(audio_data[i*frame_length:(i+1)*frame_length] ** 2))
                for i in range(num_frames)]
    energies = np.array(energies)

    if energy_threshold == "auto":
        energy_threshold = np.percentile(energies, 40)

    print(f"  Energy threshold: {energy_threshold:.4f}")

    is_speech = energies > energy_threshold
    segments, in_speech, start_frame, silence_counter = [], False, 0, 0
    min_silence_frames = int(min_silence_duration / frame_duration)

    for i, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                start_frame = i
                in_speech = True
            silence_counter = 0
        else:
            if in_speech:
                silence_counter += 1
                if silence_counter >= min_silence_frames:
                    end_frame = i - silence_counter
                    start_time = start_frame * frame_duration
                    end_time = end_frame * frame_duration
                    if end_time - start_time >= min_speech_duration:
                        segments.append({"start": round(start_time, 2),
                                         "end": round(end_time, 2)})
                    in_speech, silence_counter = False, 0

    if in_speech:
        end_time = num_frames * frame_duration
        start_time = start_frame * frame_duration
        if end_time - start_time >= min_speech_duration:
            segments.append({"start": round(start_time, 2), "end": round(end_time, 2)})

    print(f"✓ Detected {len(segments)} speech segments")

    segments_file = os.path.join(output_dir, "speech_segments.json")
    with open(segments_file, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"✓ Segments saved to: {segments_file}")
    return segments

if __name__ == "__main__":
    audio_file = "output/extracted_audio.wav"
    detect_speech_segments(audio_file)
