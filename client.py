import streamlit as st
import requests
import os

st.title("Want to add commentary to your video?")

uploaded_file = st.file_uploader('Choose a video file', type=['mp4', 'mov', 'avi', 'mkv'])
url = 'http://127.0.0.1:8000/stich_video_with_audio'

if uploaded_file is not None:
    st.success(f'File uploaded: {uploaded_file.name}')
    
    files = {'file': (uploaded_file.name, uploaded_file, 'video/mp4')}
    
    if st.button("Upload to server"):
        with st.spinner('Processing the file, please wait...'):
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            st.success("Processing complete! Downloading result...")

            # Save the video
            output_path = 'output_with_audio.mp4'
            with open(output_path, 'wb') as f:
                f.write(response.content)

            # Play the video in the app
            st.video(output_path)

            # Offer download
            with open(output_path, 'rb') as f:
                st.download_button("Download video", f, file_name="commentary_video.mp4")
        else:
            st.error(f"Server error {response.status_code}: {response.text}")
