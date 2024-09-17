# main.py

import streamlit as st

# Logo
st.image('redsquares.jpg', width=100)

# Title for the main page
st.title("Red Squares App")

# Description
st.write("""
    This is a multi-page app for music analysis. 
    Currently, you can use the following features:
    ***
    - **Key Detector:** Identify the key of a song or melody.
    - **Tuner:** Plays notes for standard guitar tuning.
    ***

    Use the sidebar to navigate to the different features.
""")


# Add a sidebar option to make it clear that users can navigate to different pages
st.sidebar.title("Navigation")
st.sidebar.write("Select a page from the options above.")
