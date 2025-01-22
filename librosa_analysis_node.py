import librosa
import numpy as np

class LibrosaAnalysisNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {
                    "multiline": False,
                    "default": "path/to/audio/file.wav",
                }),
                "analysis_type": (["default", "onset", "segment", "tempo", "mel", "spectral", "second", "half_second", "beat"], {
                    "default": "default",
                }),
                "window_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 128
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO_ENERGY", "TIMESTAMPS", "STRING", "ANALYSIS_TYPE")
    RETURN_NAMES = ("energy_levels", "timestamps", "analysis_text", "analysis_type")
    FUNCTION = "analyze_audio"
    CATEGORY = "Audio Processing"

    def analyze_audio(self, audio_file, analysis_type, window_size):
        try:
            y, sr = librosa.load(audio_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            energy_levels = []
            timestamps = []
            
            if analysis_type == "onset":
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                timestamps = librosa.frames_to_time(onset_frames, sr=sr)
                energy_levels = [librosa.feature.rms(y=y, frame_length=window_size)[0][frame] for frame in onset_frames]
                
            elif analysis_type == "segment":
                # Use librosa's segment boundaries detection
                S = np.abs(librosa.stft(y))
                segments = librosa.segment.detect_spectral_onsets(S)
                timestamps = librosa.frames_to_time(np.where(segments)[0], sr=sr)
                energy_levels = [librosa.feature.rms(y=y, frame_length=window_size)[0][frame] for frame in np.where(segments)[0]]
                
            elif analysis_type == "tempo":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo_frames = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
                timestamps = tempo_frames
                energy_levels = onset_env.tolist()
                
            elif analysis_type == "mel":
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                timestamps = librosa.frames_to_time(range(mel_spec.shape[1]), sr=sr)
                energy_levels = np.mean(mel_spec, axis=0).tolist()
                
            elif analysis_type == "spectral":
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                timestamps = librosa.frames_to_time(range(spec_cent.shape[1]), sr=sr)
                energy_levels = spec_cent[0].tolist()
                
            else:  # default, second, half_second, beat
                hop_length = window_size // 4
                energy = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)[0]
                timestamps = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length)
                
                if analysis_type == "second":
                    idx = [i for i, t in enumerate(timestamps) if round(t, 3) % 1 == 0]
                elif analysis_type == "half_second":
                    idx = [i for i, t in enumerate(timestamps) if round(t * 2, 3) % 1 == 0]
                elif analysis_type == "beat":
                    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                    timestamps = librosa.frames_to_time(beat_frames, sr=sr)
                    energy_levels = [energy[min(i, len(energy)-1)] for i in beat_frames]
                else:  # default
                    energy_levels = energy.tolist()
                    
                if analysis_type in ["second", "half_second"]:
                    energy_levels = [energy[i] for i in idx]
                    timestamps = [timestamps[i] for i in idx]
                elif analysis_type not in ["beat"]:
                    energy_levels = energy.tolist()
                    timestamps = timestamps.tolist()

            # Normalize energy levels
            energy_levels = np.array(energy_levels)
            energy_levels = (energy_levels - energy_levels.min()) / (energy_levels.max() - energy_levels.min())
            
            analysis_text = (
                f"Analysis Type: {analysis_type}\n"
                f"Duration: {duration:.2f} seconds\n"
                f"Measurements: {len(energy_levels)}\n"
                f"Window Size: {window_size}"
            )
            
            return (energy_levels.tolist(), timestamps, analysis_text, analysis_type)
            
        except Exception as e:
            return ([1.0], [0.0], f"Error: {str(e)}", "default")

NODE_CLASS_MAPPINGS = {
    "LibrosaAnalysisNode": LibrosaAnalysisNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LibrosaAnalysisNode": "Librosa Audio Analysis"
}
