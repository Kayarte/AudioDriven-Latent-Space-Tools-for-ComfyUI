import librosa
import numpy as np

class LibrosaAnalysisNode:
    """
    A node to analyze audio using Librosa with separated outputs for energy and timing data.
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
                    "default": "default",
                    "display": "Select Interval",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO_ENERGY", "TIMESTAMPS", "STRING")
    RETURN_NAMES = ("energy_levels", "timestamps", "analysis_text")
    FUNCTION = "analyze_audio"
    CATEGORY = "Audio Processing"

    def analyze_audio(self, audio_file, interval_choice):
        try:
            # Load the audio file
            y, sr = librosa.load(audio_file, sr=None)
            
            # Get audio duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Energy calculation
            hop_length = 512
            energy = np.array([
                sum(abs(y[i:i + hop_length]**2)) 
                for i in range(0, len(y), hop_length)
            ])
            energy_times = librosa.frames_to_time(
                range(len(energy)), 
                sr=sr
            )
            
            # Normalize energy to range [0, 1]
            energy = energy / np.max(energy)
            
            # Adjust intervals based on user input
            if interval_choice == "second":
                # Filter for whole seconds
                time_indices = [i for i, t in enumerate(energy_times) if round(t, 3) % 1 == 0]
                energy_times = energy_times[time_indices]
                energy = energy[time_indices]
                
            elif interval_choice == "half_second":
                # Filter for half seconds
                time_indices = [i for i, t in enumerate(energy_times) if round(t * 2, 3) % 1 == 0]
                energy_times = energy_times[time_indices]
                energy = energy[time_indices]
                
            elif interval_choice == "beat":
                # Get beat times
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                # Interpolate energy to beat times
                energy = np.interp(beat_times, energy_times, energy)
                energy_times = beat_times
            
            # Convert numpy arrays to lists and ensure they're not empty
            energy_list = energy.tolist()
            times_list = energy_times.tolist()
            
            if not energy_list or not times_list:
                raise ValueError("No energy values or timestamps were generated")
            
            # Format text output for display
            analysis_text = (
                f"Audio Duration: {duration:.2f} seconds\n"
                f"Number of Energy Measurements: {len(energy_list)}\n"
                f"Interval Type: {interval_choice}"
            )
            
            # Return tuple of all outputs
            return (energy_list, times_list, analysis_text)
            
        except Exception as e:
            # Return error state
            return ([1.0], [0.0], f"Error: {str(e)}")

# Register the node
NODE_CLASS_MAPPINGS = {
    "LibrosaAnalysisNode": LibrosaAnalysisNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LibrosaAnalysisNode": "Librosa Audio Analysis"
}
