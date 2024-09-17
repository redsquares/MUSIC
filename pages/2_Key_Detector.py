# pages/Key_Detector.py

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from st_audiorec import st_audiorec
import io
import soundfile as sf

# Logo
st.image('redsquares.jpg', width=100)

# Title for the Key Detector page
st.title("Key Detector")

# Audio input option
input_option = st.radio("Select audio input method:", ("Upload a file", "Record using microphone"))

audio_data = None
sr = None

if input_option == "Upload a file":
    # Upload audio file
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
    if audio_file is not None:
        audio_data, sr = librosa.load(audio_file, sr=None)
        st.success("Audio file uploaded successfully!")
elif input_option == "Record using microphone":
    st.write("Please record audio using the recorder below.")
    st.write("**Note:** In Safari, you may need to start and stop the recording once before it starts working properly.")
    audio_bytes = st_audiorec()

    if audio_bytes is not None:
        # Convert audio bytes to numpy array
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        st.success("Audio recorded successfully!")
    else:
        st.warning("No audio recorded yet.")

# Proceed with analysis if audio_data is available
if audio_data is not None and sr is not None and audio_data.size > 0:  # Added check for empty audio_data
    # Normalize and preprocess audio
    audio_data = audio_data.astype(float)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)  # Convert to mono
    if np.max(np.abs(audio_data)) > 0:  # Ensuring audio_data is not silent
        audio_data = audio_data / np.max(np.abs(audio_data))
    else:
        st.error("Audio data is silent.")
        st.stop()

    # Resample if necessary
    if sr != 22050:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)
        sr = 22050

    # Compute chromagram using Librosa
    chromagram = librosa.feature.chroma_cqt(y=audio_data, sr=sr)
    chroma_mean = np.mean(chromagram, axis=1)

    # Key detection including pentatonic scales
    def estimate_key(chroma_vector):
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Define profiles for different scales
        profiles = {
            'Major': np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
            'Minor': np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
            'Major Pentatonic': np.array([1, 1, 0, 0, 1, 0,
                                          1, 0, 0, 1, 0, 0]),
            'Minor Pentatonic': np.array([1, 0, 0, 1, 0, 1,
                                          0, 1, 0, 0, 1, 0])
        }

        # Normalize chroma vector
        chroma_vector = chroma_vector / np.linalg.norm(chroma_vector)

        best_key = None
        best_correlation = -np.inf
        best_scale = None

        # Iterate over all keys and scales
        for scale_name, profile in profiles.items():
            for i in range(12):
                profile_shifted = np.roll(profile, i)
                profile_norm = profile_shifted / np.linalg.norm(profile_shifted)
                correlation = np.dot(chroma_vector, profile_norm)

                if correlation > best_correlation:
                    best_correlation = correlation
                    best_key = keys[i]
                    best_scale = scale_name

        key = f"{best_key} {best_scale}"
        return key

    detected_key = estimate_key(chroma_mean)
    
    # Display the detected key
    st.write("")
    st.write(f"<h2 style='text-align: center;'>The detected key is: {detected_key}</h2>", unsafe_allow_html=True)
    st.write("")

    # Get scale notes
    def get_scale_notes(root, scale_type):
        # Note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                      'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Major and minor scale intervals
        if 'Pentatonic' in scale_type:
            if 'Major' in scale_type:
                intervals = [0, 2, 4, 7, 9]  # Major pentatonic
            else:
                intervals = [0, 3, 5, 7, 10]  # Minor pentatonic
        else:
            if 'Major' in scale_type:
                intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
            else:
                intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor scale

        root_idx = note_names.index(root)
        scale_notes = [note_names[(root_idx + interval) % 12] for interval in intervals]
        return scale_notes

    # Get the notes in the detected scale
    key_root, scale_type = detected_key.split()
    scale_notes = get_scale_notes(key_root, scale_type)

    # Visualize the piano
    st.write("### Piano")
    def draw_piano(highlight_notes):
        import matplotlib.patches as patches

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 2))

        # Piano key positions and sizes
        white_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        white_key_positions = [0, 1, 2, 3, 4, 5, 6]
        black_keys = ['C#', 'D#', 'F#', 'G#', 'A#']
        black_key_positions = [0.7, 1.7, 3.7, 4.7, 5.7]

        # Draw white keys
        for i, key in enumerate(white_keys):
            rect = patches.Rectangle((i, 0), 1, 1.5, linewidth=1, edgecolor='black',
                                     facecolor='pink' if key in highlight_notes else 'white')
            ax.add_patch(rect)
            ax.text(i + 0.5, -0.1, key, ha='center', va='top')

        # Draw black keys
        for i, key in enumerate(black_keys):
            rect = patches.Rectangle((black_key_positions[i], 0.75), 0.6, 0.75, linewidth=1, edgecolor='black',
                                     facecolor='pink' if key in highlight_notes else 'black')
            ax.add_patch(rect)

        ax.set_xlim(-0.5, 7)
        ax.set_ylim(0, 1.6)
        ax.axis('off')
        st.pyplot(fig)

    draw_piano(scale_notes)

    # General function for drawing the fretboard
    st.write("### Guitar Fretboard")
    def draw_guitar_fretboard(highlight_notes, tonic_note=None):
        import matplotlib.patches as patches

        # Define guitar strings and frets
        strings = ['E', 'A', 'D', 'G', 'B', 'E_high']  # Inverted
        num_frets = 18  # Represent 18 frets

        # Note mapping for each string and fret
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                      'F#', 'G', 'G#', 'A', 'A#', 'B']
        string_notes = {
            'E_high': [note_names[(4 + fret) % 12] for fret in range(num_frets + 1)],
            'B': [note_names[(11 + fret) % 12] for fret in range(num_frets + 1)],
            'G': [note_names[(7 + fret) % 12] for fret in range(num_frets + 1)],
            'D': [note_names[(2 + fret) % 12] for fret in range(num_frets + 1)],
            'A': [note_names[(9 + fret) % 12] for fret in range(num_frets + 1)],
            'E': [note_names[(4 + fret) % 12] for fret in range(num_frets + 1)],
        }

        fig, ax = plt.subplots(figsize=(14, 4))

        # Draw frets
        fret_positions = np.linspace(0, num_frets, num_frets + 1)
        for idx, fret_pos in enumerate(fret_positions):
            line_width = 2 if idx == 0 else 1
            ax.plot([fret_pos, fret_pos], [0, 6], color='black', linewidth=line_width)

        # Draw strings
        for i in range(7):
            ax.plot([0, fret_positions[-1]], [i, i], color='black')

        # Add string names (outside of the fretboard)
        for i, string in enumerate(strings):
            ax.text(-0.5, 6 - i, string, ha='center', va='center', fontsize=10, fontweight='bold')
            if string in highlight_notes:
                ax.text(0.5, 6 - i, string, ha='center', va='center', fontsize=10, color='pink', fontweight='bold')

        # Fret markers (updated to avoid out of bounds)
        marker_frets = [3, 5, 7, 15, 18]
        for fret in marker_frets:
            if fret < len(fret_positions) - 1:  # Avoid accessing out of bounds
                fret_pos = (fret_positions[fret] + fret_positions[fret + 1]) / 2
                ax.plot(fret_pos, 6.5, marker='o', color='black', markersize=5)

        # Double dots on 12th fret
        if 12 < len(fret_positions) - 1:
            fret_pos = (fret_positions[12] + fret_positions[13]) / 2
            ax.plot([fret_pos - 0.1, fret_pos + 0.1], [6.5, 6.5], marker='o', color='black', markersize=5)

        # Highlight notes
        for string_idx, string in enumerate(strings):
            for fret in range(num_frets):
                note = string_notes[string][fret]
                if note in highlight_notes:
                    x_pos = (fret_positions[fret] + fret_positions[fret + 1]) / 2
                    y_pos = 6 - string_idx
                    circle = patches.Circle((x_pos, y_pos), 0.4, facecolor='pink', edgecolor='black')
                    ax.add_patch(circle)
                    ax.text(x_pos, y_pos, note, fontsize=8, ha='center', va='center')

        ax.set_xlim(-1, fret_positions[-1] + 1)
        ax.set_ylim(-0.5, 7)
        ax.axis('off')
        st.pyplot(fig)

    draw_guitar_fretboard(scale_notes, key_root)

    # Visualize the chromagram
    st.write("### Chromagram")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='cool')
    plt.colorbar()
    st.pyplot(plt)

else:
    st.info("Please provide an audio input to start the analysis.")
