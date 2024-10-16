import streamlit as st
import moviepy.editor as mp
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import openai
import tempfile
import os
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa

# Initialize OpenAI API
openai.api_type = "azure"
openai.api_base = "https://internshala.openai.azure.com"
openai.api_version = "2024-08-01-preview"
openai.api_key = "22ec84421ec24230a3638d1b51e3a7dc"

def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()

    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,  # Enable word-level timestamps
    )

    response = client.recognize(config=config, audio=audio)
    
    transcript = ""
    word_timings = []
    
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "
        for word in result.alternatives[0].words:
            word_timings.append({
                'word': word.word,
                'start_time': word.start_time.total_seconds(),
                'end_time': word.end_time.total_seconds()
            })
    
    return transcript.strip(), word_timings

def correct_text(text):
    response = openai.ChatCompletion.create(
        engine="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that corrects grammatical mistakes and removes filler words."},
            {"role": "user", "content": f"Please correct the following text, removing grammatical mistakes and filler words. Maintain the overall structure and key points of the original text: {text}"}
        ]
    )
    return response.choices[0].message['content']

def text_to_speech(text):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Journey-F"
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=1.0,  # Adjustable speaking rate
        pitch=0.0  # Adjustable pitch
    )
    
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    return response.audio_content

def align_audio(original_audio, new_audio, word_timings):
    # Load audio files
    original = AudioSegment.from_wav(original_audio)
    new = AudioSegment.from_wav(new_audio)
    
    # Split the new audio on silence
    chunks = split_on_silence(new, min_silence_len=100, silence_thresh=-40)
    
    # Align chunks with word timings
    aligned_audio = AudioSegment.silent(duration=len(original))
    
    for i, chunk in enumerate(chunks):
        if i < len(word_timings):
            start_time = int(word_timings[i]['start_time'] * 1000)
            end_time = int(word_timings[i]['end_time'] * 1000)
            chunk_duration = end_time - start_time
            
            # Adjust chunk duration to match original timing
            adjusted_chunk = chunk[:chunk_duration]
            if len(adjusted_chunk) < chunk_duration:
                adjusted_chunk += AudioSegment.silent(duration=chunk_duration - len(adjusted_chunk))
            
            aligned_audio = aligned_audio.overlay(adjusted_chunk, position=start_time)
    
    return aligned_audio

def replace_audio(video_path, audio_content, word_timings):
    video = mp.VideoFileClip(video_path)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_content)
        new_audio_path = temp_audio_file.name
    
    # Extract original audio
    original_audio_path = "original_audio.wav"
    video.audio.write_audiofile(original_audio_path)
    
    # Align new audio with original
    aligned_audio = align_audio(original_audio_path, new_audio_path, word_timings)
    aligned_audio_path = "aligned_audio.wav"
    aligned_audio.export(aligned_audio_path, format="wav")
    
    # Load aligned audio and adjust duration
    audio = mp.AudioFileClip(aligned_audio_path)
    if audio.duration < video.duration:
        audio = audio.audio_loop(duration=video.duration)
    else:
        audio = audio.subclip(0, video.duration)
    
    # Apply fade in/out effects
    audio = audio.audio_fadeout(1).audio_fadein(1)
    
    final_clip = video.set_audio(audio)
    
    output_path = "output_video.mp4"
    final_clip.write_videofile(output_path)
    
    # Clean up temporary files
    os.unlink(new_audio_path)
    os.unlink(original_audio_path)
    os.unlink(aligned_audio_path)
    
    return output_path

def analyze_audio_quality(audio_path):
    y, sr = librosa.load(audio_path)
    
    # Measure signal-to-noise ratio
    snr = librosa.feature.rms(y=y).mean()
    
    # Measure spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    
    # Measure spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    
    return {
        "Signal-to-Noise Ratio": snr,
        "Spectral Centroid": centroid,
        "Spectral Bandwidth": bandwidth
    }

def main():
    st.title("Enhanced Video Audio Replacement PoC")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Process Video"):
            with st.spinner("Processing..."):
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                    temp_video_file.write(uploaded_file.read())
                    video_path = temp_video_file.name
                
                # Extract audio from video
                video = mp.VideoFileClip(video_path)
                audio = video.audio
                audio_path = "temp_audio.wav"
                audio.write_audiofile(audio_path)
                
                # Analyze original audio quality
                original_quality = analyze_audio_quality(audio_path)
                st.subheader("Original Audio Quality")
                st.json(original_quality)
                
                # Transcribe audio
                transcription, word_timings = transcribe_audio(audio_path)
                st.subheader("Original Transcription:")
                st.write(transcription)
                
                # Correct transcription
                corrected_text = correct_text(transcription)
                st.subheader("Corrected Transcription:")
                st.write(corrected_text)
                
                # Generate new audio
                new_audio_content = text_to_speech(corrected_text)
                
                # Replace audio in video
                output_path = replace_audio(video_path, new_audio_content, word_timings)
                
                # Analyze new audio quality
                new_quality = analyze_audio_quality(output_path)
                st.subheader("New Audio Quality")
                st.json(new_quality)
                
                st.success("Video processed successfully!")
                st.video(output_path)
                
                # Clean up temporary files
                os.unlink(video_path)
                os.unlink(audio_path)
                os.unlink(output_path)

if __name__ == "__main__":
    main()