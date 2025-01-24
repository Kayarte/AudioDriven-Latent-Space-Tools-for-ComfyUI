Librosa Audio Analysis to Noise node for ComfyUI.

This is a custom node for ComfyUI that analyzes audio files using Librosa, extracting tempo, beat times, energy levels, and timestamps. The analysis results can be displayed in a text box within ComfyUI.

Features

    Audio Duration: Shows the total length of the audio in seconds.
    Tempo: Displays the beats per minute (BPM) of the audio.
    Beat Times: Provides timestamps for each beat detected in the audio.
    Energy Levels: Displays the energy levels calculated at intervals throughout the audio.
    Energy Timestamps: Lists the corresponding timestamps for each energy level.

Settings

Average measurements by analysis type (from 3s, 11.65s, and 89s samples):
WARNING: Typical frame counts

    default: huge (~122K for 89s)
    
    mel/spectral/tempo: high (~7.6K for 89s)

     onset: moderate (60-500 depending on complexity) 

     beat: low (14-168 based on tempo) 

     second: consistent (~length in seconds) 

     half_second: consistent (~2x seconds) 


Choose analysis type based on desired frame count and use case.

Default, onset, *segment*, tempo, Mel (my fav.), spectral, second, half second, beat. 

*Librosa Audio Analysis - settings = segment doesnt work.* 

Gaussian, salt and pepper, Perlin.
        
Installation

Download or clone this repository to your ComfyUI\custom_nodes folder

For a Greater Project.
This node is part of a ongoing solo project to integrate music analysis.

![image](https://github.com/user-attachments/assets/f84ee035-968f-4e8b-b9e7-ccc1c45a92c8)

