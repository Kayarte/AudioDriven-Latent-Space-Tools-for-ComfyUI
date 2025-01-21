import librosa
import numpy as np

class LibrosaAnalysisNode:
    """
    A node to analyze audio using Librosa and display results in a text box with customizable settings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {
                    "multiline": False,
                    "default": "path/to/audio/file.wav",
                }),
                "interval_choice": (["default", "second", "half_second", "beat"], {
                    "default": "default",  # Options: "default", "second", "half_second", "beat"
                    "display": "Select Interval",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)  # Text output
    RETURN_NAMES = ("analysis_output",)
    FUNCTION = "analyze_audio"
    CATEGORY = "Audio Processing"

    def analyze_audio(self, audio_file, interval_choice):
        try:
            # Load the audio file
            y, sr = librosa.load(audio_file, sr=None)

            # Get audio duration
            duration = librosa.get_duration(y=y, sr=sr)

            # Beat detection
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

            # Energy calculation
            hop_length = 512
            energy = np.array([
                sum(abs(y[i:i + hop_length]**2)) for i in range(0, len(y), hop_length)
            ])
            energy_times = librosa.frames_to_time(range(len(energy)), sr=sr).tolist()
            energy = energy / max(energy)  # Normalize the energy levels

            # Adjust energy intervals based on user input
            if interval_choice == "second":
                energy_times = [t for t in energy_times if t % 1 == 0]  # Filter out non-whole second intervals
                beat_times = [t for t in beat_times if t % 1 == 0]      # Filter beat times for each second
            elif interval_choice == "half_second":
                energy_times = [t for t in energy_times if t % 0.5 == 0]  # Filter out non-half second intervals
                beat_times = [t for t in beat_times if t % 0.5 == 0]     # Filter beat times for each half second
            elif interval_choice == "beat":
                # Keep the energy times and beat times based on actual beats
                energy_times = beat_times  # Sync energy times with beat times
                energy = energy[:len(beat_times)]  # Align energy with beat counts

            # Format output without rounding
            result = (
                f"Audio Duration: {duration:.2f} seconds\n"
                f"Tempo: {tempo} BPM\n"
                f"Total Beat Times: {len(beat_times)}\n"
                f"Beat Times ({interval_choice}): {beat_times}\n"
                f"Energy Levels ({interval_choice}): {energy.tolist()}\n"
                f"Energy Timestamps ({interval_choice}): {energy_times}"
            )

        except Exception as e:
            result = f"Error: {str(e)}"

        return (result,)

# Register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "LibrosaAnalysisNode": LibrosaAnalysisNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LibrosaAnalysisNode": "Librosa Audio Analysis"
}
