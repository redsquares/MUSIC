# pages/Key_Detector.py

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from st_audiorec import st_audiorec
import io
import soundfile as sf

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
    st.write("### Guitar")

    def draw_guitar_fretboard(highlight_notes):
        import matplotlib.patches as patches

        # Define guitar strings and frets
        strings = ['E', 'A', 'D', 'G', 'B', 'E_high']
        num_frets = 18  # Represent 18 frets

        # Note mapping for each string and fret
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        string_notes = {
            'E_high': [note_names[(4 + fret) % 12] for fret in range(num_frets + 1)],
            'B': [note_names[(11 + fret) % 12] for fret in range(num_frets + 1)],
            'G': [note_names[(7 + fret) % 12] for fret in range(num_frets + 1)],
            'D': [note_names[(2 + fret) % 12] for fret in range(num_frets + 1)],
            'A': [note_names[(9 + fret) % 12] for fret in range(num_frets + 1)],
            'E': [note_names[(4 + fret) % 12] for fret in range(num_frets + 1)],
        }

        fig, ax = plt.subplots(figsize=(14, 4))

        # Draw vertical lines for frets (comb) with thicker lines
        fret_positions = np.linspace(1, num_frets, num_frets)  # Start at 1 to represent fret 1
        for idx, fret_pos in enumerate(fret_positions):
            line_width = 3 if idx == 0 else 1  # Thicker line for the comb's start
            ax.plot([fret_pos, fret_pos], [0, 6], color='black', linewidth=line_width)

        # Draw horizontal lines for strings, shortened to start from the comb
        for i in range(6):  # 6 strings
            ax.plot([0.5, num_frets], [i, i], color='black', linewidth=1)  # Start from 0.5 to align with comb

        # Draw the top horizontal line
        ax.plot([0.5, num_frets], [6, 6], color='black', linewidth=1)  # Top line

        # Fret markers
        marker_frets = [3, 5, 7, 12, 15, 18]
        for fret in marker_frets:
            if fret < len(fret_positions):  # Avoid out-of-bounds index
                fret_pos = (fret_positions[fret - 1] + fret_positions[fret]) / 2
                if fret == 12:
                    # Double dots for fret 12
                    ax.plot([fret_pos - 0.1, fret_pos + 0.1], [6.5, 6.5], marker='o', color='black', markersize=5)
                else:
                    ax.plot(fret_pos, 6.5, marker='o', color='black', markersize=5)

        # Highlight notes (align labels with spaces)
        for string_idx, string in enumerate(strings):
            for fret in range(1, num_frets + 1):  # Start from fret 1
                if fret < num_frets:  # Avoid accessing out-of-bounds index
                    note = string_notes[string][fret]
                    if note in highlight_notes:
                        x_pos = (fret_positions[fret - 1] + fret_positions[fret]) / 2
                        y_pos = 6 - string_idx - 0.5  # Align notes in spaces
                        circle_color = 'lightgreen' if note == key_root else 'pink'  # Change tonic note color
                        circle = patches.Circle((x_pos, y_pos), 0.4, facecolor=circle_color)  # Removed outer border
                        ax.add_patch(circle)
                        ax.text(x_pos, y_pos, note, fontsize=8, ha='center', va='center')

        # Add open string notes (3 pixels to the left)
        for i, string in enumerate(strings):
            open_note = string_notes[string][0]
            y_pos = 6 - i - 0.5  # Original height for open strings
            circle_color = 'lightgreen' if open_note == key_root else 'pink'  # Change tonic note color
            ax.add_patch(patches.Circle((0.47, y_pos), 0.4, color=circle_color))  # Move circles 3 pixels to the left
            ax.text(0.47, y_pos, open_note, fontsize=8, ha='center', va='center')  # Align with circles

        ax.set_xlim(-1.5, fret_positions[-1] + 1)
        ax.set_ylim(-0.5, 7)
        ax.axis('off')
        st.pyplot(fig)

    
    draw_guitar_fretboard(scale_notes)

    # Visualize the chromagram
    st.write("### Chromagram")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='cool')
    plt.colorbar()
    st.pyplot(plt)

else:
    st.info("Please provide an audio input to start the analysis.")
