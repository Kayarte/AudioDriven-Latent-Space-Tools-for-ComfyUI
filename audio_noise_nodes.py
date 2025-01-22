import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class TimeBase(Enum):
    BEAT = "beat"
    SECOND = "second"
    HALF = "half"
    SPECTRAL = "spectral"

@dataclass
class NoiseParams:
    intensity: float
    grain: float
    persistence: float
    distribution: str

class AudioNoiseMapper:
    """Maps audio features to noise parameters for image generation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "energy_levels": ("AUDIO_ENERGY",),
                "timestamps": ("TIMESTAMPS",),
                "time_base": (["beat", "second", "half", "spectral"],),
            }
        }
    
    RETURN_TYPES = ("NOISE_PARAMS", "STRING")
    RETURN_NAMES = ("noise_params", "debug_info")
    FUNCTION = "process_energy_to_noise"
    CATEGORY = "audio/noise"

    def __init__(self):
        self._setup_time_base_params()
    
    def _setup_time_base_params(self):
        self.params = {
            "beat": {
                "smoothing": 0.2,
                "scale": 1.0,
                "window_size": 4
            },
            "second": {
                "smoothing": 0.1,
                "scale": 2.0,
                "window_size": 1
            },
            "half": {
                "smoothing": 0.3,
                "scale": 0.5,
                "window_size": 2
            },
            "spectral": {
                "smoothing": 0.05,
                "scale": 1.5,
                "window_size": 8
            }
        }
    
    def process_energy_to_noise(self, 
                              energy_levels: List[float],
                              timestamps: List[float],
                              time_base: str) -> Tuple[Dict[str, NoiseParams], str]:
        try:
            # Ensure inputs are valid
            if not energy_levels or not timestamps:
                raise ValueError("Empty energy levels or timestamps received")
            
            # Get parameters for selected time base
            params = self.params[time_base]
            
            # Convert to numpy arrays
            energy_array = np.array(energy_levels)
            
            # Smooth the energy data
            smooth_energy = self._smooth_signal(energy_array, params["window_size"])
            
            # Generate different noise distributions
            noise_params = {
                "gaussian": {
                    "intensity": float(np.mean(smooth_energy) * params["scale"] * 3),
                    "grain": float(np.std(smooth_energy) * 5),
                    "persistence": float(self._calculate_persistence(smooth_energy, params)),
                    "distribution": "gaussian"
                },
                "salt_pepper": {
                    "intensity": float(np.sum(smooth_energy > np.percentile(smooth_energy, 75)) / len(smooth_energy)),
                    "grain": float(1.0 / params["window_size"]),
                    "persistence": float(self._calculate_persistence(smooth_energy, params)),
                    "distribution": "salt_pepper"
                },
                "perlin": {
                    "intensity": float(np.max(smooth_energy) * params["scale"] * 2),
                    "grain": float(1.0 / (np.std(smooth_energy) + 1e-6)),
                    "persistence": float(self._calculate_persistence(smooth_energy, params)),
                    "distribution": "perlin"
                },
                "timestamps": timestamps
            }

            # Debugging prints
            print(f"Received noise_params: {noise_params}")
            print(f"Gaussian params: {noise_params.get('gaussian')}")

            debug_info = (
                f"Processed {len(energy_levels)} energy values\n"
                f"Time base: {time_base}\n"
                f"Window size: {params['window_size']}\n"
                f"Scale: {params['scale']}"
            )
            
            return (noise_params, debug_info)
            
        except Exception as e:
            return ({}, f"Error in noise mapping: {str(e)}")
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        window = np.ones(window_size) / window_size
        return np.convolve(signal, window, mode='same')
    
    def _calculate_persistence(self, energy: np.ndarray, params: Dict) -> float:
        diff = np.diff(energy)
        return float(np.exp(-np.std(diff) * params["scale"]))

class RhythmicNoiseModulator:
    """Modulates noise parameters based on rhythmic features"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_params": ("NOISE_PARAMS",),
                "beat_times": ("BEAT_TIMES",),
                "energy_levels": ("AUDIO_ENERGY",),
                "beat_division": ("INT", {"default": 4, "min": 1, "max": 16}),
            }
        }
    
    RETURN_TYPES = ("MODULATED_NOISE_PARAMS",)
    FUNCTION = "modulate_noise"
    CATEGORY = "audio/noise"

    def modulate_noise(self,
                      noise_params: Dict[str, NoiseParams],
                      beat_times: List[float],
                      energy_levels: List[float],
                      beat_division: int) -> Tuple[List[NoiseParams]]:
        """Create rhythmically modulated noise parameters"""
        
        # Create subdivided beat grid
        beat_grid = self._create_beat_grid(beat_times, beat_division)
        
        # Generate modulation envelopes
        intensity_mod = self._create_intensity_envelope(beat_grid, energy_levels)
        grain_mod = self._create_grain_envelope(beat_grid, energy_levels)
        
        # Apply modulation to each noise type
        modulated_params = {}
        for noise_type, base_params in noise_params.items():
            if noise_type != "timestamps":  # Skip the timestamps entry
                mod_params = []
                for i in range(len(beat_grid)):
                    mod_param = NoiseParams(
                        intensity=base_params.intensity * intensity_mod[i],
                        grain=base_params.grain * grain_mod[i],
                        persistence=base_params.persistence,
                        distribution=base_params.distribution
                    )
                    mod_params.append(mod_param)
                modulated_params[noise_type] = mod_params
        
        modulated_params["timestamps"] = beat_grid.tolist()
        return (modulated_params,)
    
    def _create_beat_grid(self, beat_times: List[float], beat_division: int) -> np.ndarray:
        grid = []
        for i in range(len(beat_times) - 1):
            interval = beat_times[i+1] - beat_times[i]
            subdivisions = np.linspace(0, interval, beat_division, endpoint=False)
            grid.extend(beat_times[i] + subdivisions)
        return np.array(grid)
    
    def _create_intensity_envelope(self, grid: np.ndarray, energy: List[float]) -> np.ndarray:
        return np.interp(grid, np.arange(len(energy)), energy)
    
    def _create_grain_envelope(self, grid: np.ndarray, energy: List[float]) -> np.ndarray:
        energy_diff = np.diff(energy, prepend=energy[0])
        return 1.0 + np.abs(np.interp(grid, np.arange(len(energy_diff)), energy_diff))

# Register the nodes
NODE_CLASS_MAPPINGS = {
    "AudioNoiseMapper": AudioNoiseMapper,
    "RhythmicNoiseModulator": RhythmicNoiseModulator,
}

# Add descriptions for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioNoiseMapper": "Audio To Noise Parameters",
    "RhythmicNoiseModulator": "Rhythmic Noise Modulator"
}
