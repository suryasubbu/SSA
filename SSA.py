
from pathlib import Path
from pydub import AudioSegment
from yt_dlp import YoutubeDL
import pandas as pd
import numpy as np
import whisper 
from stable_whisper import modify_model
import os
import requests
import json
import time
import librosa
import config as config

def clupping(a):
    grouped_segments = []
    current_group = []
    current_duration = 0.0
    a['start_diff'] = a['start'].diff().shift(-1)
    a['start_diff'] = a['start_diff'].fillna(a["total"])
    # Iterate over the dataframe rows
    for idx, row in a.iterrows():
        if current_duration < config.threshold :

            if current_duration + row['start_diff'] > config.threshold:
                # Combine the text of the current group
                combined_text = ' '.join([segment['text'] for segment in current_group])
                
                # Create a new row for the grouped segments
                new_row = {
                    'Segment no': [segment['Segment no'] for segment in current_group],
                    'start': current_group[0]['start'],
                    'end': current_group[-1]['end'],
                    'text': combined_text,
                    'total': current_duration,
                    'word_count': sum(segment['word_count'] for segment in current_group)
                }
                
                # Add the new row to the list of grouped segments
                grouped_segments.append(new_row)
                
                # Reset the current group and duration
                current_group = []
                current_duration = 0.0
        
        else:
                
                # Create a new row for the grouped segments
                new_row = {
                    'Segment no': row["Segment no"],
                    'start': row['start'],
                    'end': row['end'],
                    'text': row["text"],
                    'total': row["total"],
                    'word_count':  row["word_count"]
                }
                
                # Add the new row to the list of grouped segments
                grouped_segments.append(new_row)
                
                # Reset the current group and duration
                current_group = []
                current_duration = 0.0
        
        # Add the current row to the group
        current_group.append(row)
        current_duration += row['start_diff']

    # If there are any remaining rows in the current group, add them as a final segment
    if current_group:
        combined_text = ' '.join([segment['text'] for segment in current_group])
        new_row = {
            'Segment no': [segment['Segment no'] for segment in current_group],
            'start': current_group[0]['start'],
            'end': current_group[-1]['end'],
            'text': combined_text,
            'total': current_duration,
            'word_count': sum(segment['word_count'] for segment in current_group)
        }
        grouped_segments.append(new_row)

    # Convert the list of grouped segments to a new dataframe
    grouped_df = pd.DataFrame(grouped_segments)

    return grouped_df


def word_count(row):
    return len(row.split())

def isMusic(path):
        y, sr = librosa.load(path, sr = None)
        splits = librosa.effects.split(y = y, frame_length = 500, top_db = 10)

        if splits.size <= 3:
                return True
        else:
                return False
        
def rem_music(d_path):
     cc = []
     data_dir = d_path
     file_paths = pd.DataFrame({
     "path": [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.lower().endswith('.wav') and os.path.isfile(os.path.join(data_dir, filename))]
     })
     for index, row in file_paths.iterrows():
        try:
            if isMusic(row["path"]):
                print(f"Is Music: {isMusic(row["path"])}",row["path"])
                cc.append = row["path"]
                # os.remove(row["path"])
        except:
             continue
     print(cc)
def sentence_splitter(input_file,i,out_dir = f"output/{config.speaker_name}/"):
    output_directory = out_dir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    model = whisper.load_model('medium.en')
    # result1 = model.transcribe('/home/suryasss/Barack Obama_ Yes We Can.mp3', language='en', max_initial_timestamp=None)
    modify_model(model)
    result2 = model.transcribe(input_file, language='en') 
    a = result2.segments
    data = {
    'Segment no': range(0, len(a)),
    'start': [segment.start for segment in a],
    'end': [segment.end for segment in a],
    'text': [segment.text for segment in a]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df["end"] = df.end + 0.25
    df["total"] = df.end - df.start
    df['word_count'] = df['text'].apply(word_count)
    df = clupping(df)
    fil_df = df[df.total > (config.threshold/10)][df.word_count >= 6].reset_index(drop = True)
    j = i+len(fil_df["end"])
    b = np.arange(i,j)
    fil_df["inde"] = b
    fil_df['filename'] = fil_df['inde'].apply(lambda x: f'{config.speaker_name}_{x:05d}.wav')
    

    for index, row in fil_df.iterrows():
        end_time = row['end']
        start_time = row['start']
        duration = end_time - start_time  # Calculate duration of the chunk
        output_file = os.path.join(out_dir, f"{config.speaker_name}_{i}.wav")  # Output file path for each chunk
        
        chunk =  AudioSegment.silent(duration = 100) + AudioSegment.from_file(input_file)[start_time * 1000:(end_time * 1000)] + AudioSegment.silent(duration = 100)

        chunk.export(output_file, format="wav")
        i = i+1    
    fil_df[["filename","text","start","end"]].to_csv(f'speaker_audios/{config.speaker_name}/meta_chunk.csv',index = False)
    os.remove(input_file)
    return i,fil_df[["filename","text","start","end"]]

def convert_webm_to_wav(input_file, output_file):
    # Load the WebM file using pydub
    audio = AudioSegment.from_file(input_file, format='webm')

    # Export the audio as WAV format
    audio.export(output_file, format='wav')

def split_audio_by_speaker(input_audio_file, segments, out_dir = f'speaker_audios/{config.speaker_name}/'):
    # Load the input audio using pydub
    audio = AudioSegment.from_file(input_audio_file)
    audio = audio.set_channels(1)

    # Create an output directory
    output_directory = out_dir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create a dictionary to hold speaker segments
    speaker_segments = {}

    # Group segments by speaker
    for i, (speaker, start_time, end_time) in enumerate(segments):
        # If the speaker is not already in the dictionary, create a new list
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []

        # Append the segment to the list of segments for the speaker
        speaker_segments[speaker].append((start_time, end_time))

    concatenated = AudioSegment.empty()
    # Split the audio file into speaker segments
    for speaker, segments in speaker_segments.items():
        # Sort the segments by start time
        segments.sort(key=lambda x: x[0])

        # Generate the output filename
        output_file = os.path.join(output_directory, f"speaker_{speaker}.wav")

        # Concatenate the segments and export as a single audio file
        concatenated = AudioSegment.empty()
        for start_time, end_time in segments:
            segment = audio[start_time:end_time]
            concatenated = concatenated + segment

        concatenated.export(output_file, format="wav")
    return concatenated

def download_audio(url, start_time, out_filename):

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ss', start_time,
            '-ar', '16000'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'outtmpl': out_filename.split('.')[0]
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # os.remove(out_filename + '.wav')


def speaker_diarization(wav_filename, out_dir = f'speaker_audios/{config.speaker_name}/'):

    ## Use the transcribe function from audio.py 
    # Upload the WAV file to AssemblyAI for transcription
    base_url = "https://api.assemblyai.com/v2"
    headers = {"authorization": "e2ce4a6e07c745668be2468dd9a34d30"}

    with open(wav_filename, "rb") as f:
        response = requests.post(base_url + "/upload", headers=headers, data=f)

    upload_url = response.json()["upload_url"]
    data = {"audio_url": upload_url, "speaker_labels": True}
    url = base_url + "/transcript"
    response = requests.post(url, json=data, headers=headers)
    transcript_id = response.json()['id']
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()
        transcription_segments = []

        if transcription_result['status'] == 'completed':
            if 'utterances' in transcription_result:
                utterances = transcription_result['utterances']
                for utterance in utterances:
                    speaker = utterance['speaker']
                    start = utterance['start']
                    end = utterance['end']
                    transcription_segments.append((speaker, start, end))

                # Print the transcription segments
                for segment in transcription_segments:
                    print(segment)

                # Split the audio by speaker
                split_audio_by_speaker(wav_filename, transcription_segments, out_dir=out_dir)
            else:
                print("No utterances found in the transcription.")
            break

        elif transcription_result['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")