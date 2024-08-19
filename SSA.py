
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
from pydub import AudioSegment
def segment_audio(file_path, segment_duration,i, out_dir = f"{config.out_dir}/{config.speaker_name}/"):
    """
    Segments a given wav file into chunks of specified duration and saves them with unique filenames.

    Args:
        file_path (str): Path to the input wav file.
        segment_duration (int): Duration of each segment in seconds.
        output_dir (str): Directory where the segmented wav files will be saved.
    
    Returns:
        None
    """
    model2 = whisper.load_model('medium.en')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    audio = AudioSegment.from_wav(file_path)
    segment_duration_ms = segment_duration * 1000  # pydub works in milliseconds

    total_segs = len(audio) // segment_duration_ms
    texted = []

    for j in range(total_segs):
        start_time = i * segment_duration_ms
        end_time = start_time + segment_duration_ms
        audio_chunk = audio[start_time:end_time]
        output_file_name = f"{config.speaker_name}_{i:05d}.wav"
        output_file = os.path.join(out_dir, f"{config.speaker_name}_{i:05d}.wav")
        audio_chunk.export(output_file, format="wav")
        text = model2.transcribe(output_file,language='en')
        texted.append([output_file_name,text["text"]])

        print(f"Segment {i+1} saved as {output_file_name}")
        i=i+1

    meta = pd.DataFrame(texted)
    meta.columns = ["filename","transcripts"]
    return i,meta


def parse_time(timestamp):
    """
    Converts a timestamp in HH:MM:SS format to seconds.
    
    Parameters:
    - timestamp: str, time in HH:MM:SS format.
    
    Returns:
    - total_seconds: float, total time in seconds.
    """
    h, m, s = map(float, timestamp.split(':'))
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds

def trim_and_overwrite_audio(input_file, start_time_str):
    """
    Trims the audio file from a given start time to the end and overwrites the original file.
    
    Parameters:
    - input_file: str, path to the input WAV file.
    - start_time_str: str, start time in HH:MM:SS format where the trim should begin.
    """
    # Parse the start time
    start_time = parse_time(start_time_str)
    
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    
    # Calculate the start time in milliseconds
    start_time_ms = start_time * 1000
    
    # Trim the audio from start_time to the end
    trimmed_audio = audio[start_time_ms:]
    
    # Overwrite the original file with the trimmed audio
    trimmed_audio.export(input_file, format="wav")
    print(f"Audio trimmed and original file overwritten")
def clupping_voiced(a):
    grouped_segments = []
    current_group = []
    current_duration = 0.0
    a['start_diff'] = a['start'].diff().shift(-1)
    a['start_diff'] = a['start_diff'].fillna(a["total"])
    # print(a)
    # Iterate over the dataframe rows
    for idx, row in a.iterrows():
        try:
            if current_duration < config.threshold :

                if current_duration + row['start_diff'] > config.threshold:
                    # Combine the text of the current group
                    combined_text = ' '.join([segment['text'] for segment in current_group])
                    # print(combined_text)
                    # Create a new row for the grouped segments
                    new_row = {
                        'segment_no': [segment['segment_no'] for segment in current_group],
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
                        'segment_no': row["segment_no"],
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
        except:
            continue
        
        # Add the current row to the group
        current_group.append(row)
        current_duration += row['start_diff']

    # If there are any remaining rows in the current group, add them as a final segment
    if current_group:
        combined_text = ' '.join([segment['text'] for segment in current_group])
        new_row = {
            'segment_no': [segment['segment_no'] for segment in current_group],
            'start': current_group[0]['start'],
            'end': current_group[-1]['end'],
            'text': combined_text,
            'total': current_duration,
            'word_count': sum(segment['word_count'] for segment in current_group)
        }
        grouped_segments.append(new_row)

    # Convert the list of grouped segments to a new dataframe
    grouped_df = pd.DataFrame(grouped_segments)
    print(grouped_df)

    return grouped_df

def clupping(a):
    grouped_segments = []
    current_group = []
    current_duration = 0.0
    a['start_diff'] = a['start'].diff().shift(-1)
    a['start_diff'] = a['start_diff'].fillna(a["total"])
    print(a)
    # Iterate over the dataframe rows
    for idx, row in a.iterrows():
        try:
            if current_duration < config.threshold :

                if current_duration + row['start_diff'] > config.threshold:
                    # Combine the text of the current group
                    combined_text = ' '.join([segment['text'] for segment in current_group])
                    print(combined_text)
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
        except:
            continue
        
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
        splits = librosa.effects.split(y = y, frame_length = 500, top_db = 5)

        if splits.size <= 5:
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
                os.remove(row["path"])
                print("removed",row["path"])
        except:
             continue
     print(cc)
    
def sentence_splitter_voiced(input_file,i,out_dir = f"{config.out_dir}/{config.speaker_name}/"):

    threshold = config.threshold
    speaker_name = config.speaker_name
    output_directory = out_dir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    model = whisper.load_model('large-v2')
    model2 = whisper.load_model('medium.en')
    # result1 = model.transcribe('/home/suryasss/Barack Obama_ Yes We Can.mp3', language='en', max_initial_timestamp=None)
    modify_model(model)
    result2 = model.transcribe(input_file, language='en') 
    a = result2.segments
    # print(a)
    data = {
    'segment_no': range(0, len(a)),
    'start': [segment.start for segment in a],
    'end': [segment.end for segment in a],
    'text': [segment.text for segment in a]
    }

    # Create DataFrame
    df = pd.DataFrame(data)


    # Load the original audio file
    original_audio = AudioSegment.from_file(input_file)

    # Initialize the final audio segment
    final_audio = AudioSegment.silent(duration=0)

    # Define the duration for non-segment to be added (1 second)
    non_segment_duration = 0.2 * 1000  # pydub works with milliseconds

    # Initialize a list to store updated segment times
    updated_times = []

    # Track the current position in the final_audio
    current_position = 0
    df["shifted_start"] = df["start"].shift(-1).fillna(df["end"])
    df["end"] = np.where((df["end"] + 0.3) >= df["shifted_start"],df["shifted_start"],df["end"] + 0.3)
    # Append the first segment
    first_segment_start_time = df.iloc[0]['start'] * 1000  # Convert to milliseconds
    first_segment_end_time = df.iloc[0]['end'] * 1000  # Convert to milliseconds
    first_segment = original_audio[first_segment_start_time:first_segment_end_time]
    final_audio += first_segment


    updated_times.append({
        'segment_no': df.iloc[0]['segment_no'],
        'start': current_position / 1000,  # Convert back to seconds
        'end': (current_position + len(first_segment)) / 1000  # Convert back to seconds
    })

    # Update current position
    current_position += len(first_segment)

    # Loop through each pair of segments
    for t in range(1, len(df)):
        prev_segment_end_time = df.iloc[t-1]['end'] * 1000  # End of previous segment
        current_segment_start_time = df.iloc[t]['start'] * 1000  # Start of current segment
        current_segment_end_time = df.iloc[t]['end'] * 1000  # End of current segment
        
        # Calculate the duration of gap to use
        gap_duration = min(non_segment_duration, current_segment_start_time - prev_segment_end_time)
        
        # Add the non-segment part
        if gap_duration > 0.5:
            non_segment = original_audio[prev_segment_end_time:prev_segment_end_time + gap_duration]
        else:
            non_segment = AudioSegment.silent(duration=0.5)
        final_audio += non_segment
        current_position += len(non_segment)
        
        # Add the current segment
        current_segment = original_audio[current_segment_start_time:current_segment_end_time]
        final_audio += current_segment

        
        # Store the updated times
        updated_times.append({
            'segment_no': df.iloc[t]['segment_no'],
            'start': current_position / 1000,  # Convert back to seconds
            'end': (current_position + len(current_segment)) / 1000  # Convert back to seconds
        })
        
        # Update current position
        current_position += len(current_segment)
    total_file = f"{config.out_dir}/speaker_audios/{config.speaker_name}/less_voiced{i}.wav"
    final_audio.export(total_file, format="wav")
    # Create a new DataFrame with updated times
    updated_df = pd.DataFrame(updated_times)
    updated_df["text"] = df["text"]
    updated_df["total"] = updated_df.end - updated_df.start
    updated_df['word_count'] = updated_df['text'].apply(word_count) 
    adf = clupping_voiced(updated_df)
    print(adf.columns)
    fil_df = adf[adf.total > (threshold/1.75)][adf.word_count >= 6].reset_index(drop = True)
    j = i+len(fil_df["end"])
    b = np.arange(i,j)
    fil_df["inde"] = b
    fil_df['filename'] = fil_df['inde'].apply(lambda x: f'{speaker_name}_{x:05d}.wav')
    print(fil_df.shape)
    is_music = []
    texted = []
    for index, row in fil_df.iterrows():
        end_time = row['end']
        start_time = row['start']
        duration = end_time - start_time  # Calculate duration of the chunk
        output_file = os.path.join(out_dir, f"{speaker_name}_{i:05d}.wav")  # Output file path for each chunk
        
        chunk =  final_audio[start_time * 1000:(end_time * 1000)]
        
        # print(isMusic(chunk))
        chunk.export(output_file, format="wav")
        text = model2.transcribe(output_file,language='en')
        texted.append(text["text"])
        is_music.append(isMusic(output_file))
        i = i+1 
    fil_df["ismusic"]   = is_music 
    fil_df["transcript"]    = texted
    fil_df.drop_duplicates("transcript",inplace = True)
    print("after dropping duplicates",fil_df.shape)
    fil_df[["filename","text","transcript","start","end","ismusic"]].to_csv(f'{out_dir}/meta_chunk_{i-1}.csv',index = False)
    return i,fil_df[["filename","transcript"]]
    # return df
def sentence_splitter(input_file,i,out_dir = f"{config.out_dir}/{config.speaker_name}/"):
    output_directory = out_dir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    model = whisper.load_model('small.en')
    model2 = whisper.load_model('large-v2')
    # result1 = model.transcribe('/home/suryasss/Barack Obama_ Yes We Can.mp3', language='en', max_initial_timestamp=None)
    modify_model(model)
    result2 = model.transcribe(input_file, language='en') 
    a = result2.segments
    # print(a)
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
    fil_df = df[df.total > (config.threshold/1.75)][df.word_count >= 6].reset_index(drop = True)
    j = i+len(fil_df["end"])
    b = np.arange(i,j)
    fil_df["inde"] = b
    fil_df['filename'] = fil_df['inde'].apply(lambda x: f'{config.speaker_name}_{x:05d}.wav')
    print(fil_df.shape)
    is_music = []
    texted = []
    for index, row in fil_df.iterrows():
        end_time = row['end']
        start_time = row['start']
        duration = end_time - start_time  # Calculate duration of the chunk
        output_file = os.path.join(out_dir, f"{config.speaker_name}_{i:05d}.wav")  # Output file path for each chunk
        
        chunk =  AudioSegment.from_file(input_file)[start_time * 1000:(end_time * 1000)]
        
        # print(isMusic(chunk))
        chunk.export(output_file, format="wav")
        text = model2.transcribe(output_file,language='en')
        texted.append(text["text"])
        is_music.append(isMusic(output_file))
        i = i+1 
    fil_df["ismusic"]   = is_music 
    fil_df["texted"]    = texted
    fil_df.drop_duplicates("texted",inplace = True)
    print("after dropping duplicates",fil_df.shape)
    fil_df[["filename","text","texted","start","end","ismusic"]].to_csv(f'{config.out_dir}/{config.speaker_name}/meta_chunk_{i-1}.csv',index = False)
    return i,fil_df[["filename","texted","start","end"]]

def convert_webm_to_wav(input_file, output_file):
    # Load the WebM file using pydub
    audio = AudioSegment.from_file(input_file, format='webm')

    # Export the audio as WAV format
    audio.export(output_file, format='wav')

def split_audio_by_speaker(input_audio_file, segments, out_dir = f'{config.out_dir}/speaker_audios/{config.speaker_name}/'):
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
        # print(segments)
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
        concatenated = AudioSegment.silent(duration = 100)
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
            '-ss', "00:00:00",
            '-ar', '16000'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'outtmpl': out_filename.split('.')[0]
    }

    with YoutubeDL(ydl_opts) as ydl:
        # print(url)
        ydl.download([url])
    o_n = out_filename.split('.')[0] + ".wav"
    print("******************************************************")
    print("trimming")
    trim_and_overwrite_audio(o_n, str(start_time))

    # os.remove(out_filename + '.wav')


def speaker_diarization(wav_filename, out_dir = f'{config.out_dir}/speaker_audios/{config.speaker_name}/'):

    ## Use the transcribe function from audio.py 
    # Upload the WAV file to AssemblyAI for transcription
    base_url = "https://api.assemblyai.com/v2"
    headers = {"authorization": "52cacc23d62d4ddb83ab8de8109e1d68"}

    with open(wav_filename, "rb") as f:
        response = requests.post(base_url + "/upload", headers=headers, data=f)

    upload_url = response.json()["upload_url"]
    data = {"audio_url": upload_url, "speaker_labels": True}
    url = base_url + "/transcript"
    response = requests.post(url, json=data, headers=headers)
    # print(response.json())
    transcript_id = response.json()['id']
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()
        transcription_segments = []

        if transcription_result['status'] == 'completed':
            if 'utterances' in transcription_result:
                utterances = transcription_result['utterances']
                # print(utterances)
                for utterance in utterances:
                    speaker = utterance['speaker']
                    start = utterance['start']
                    end = utterance['end']
                    transcription_segments.append((speaker, start, end))

                # Print the transcription segments
                # for segment in transcription_segments:
                    # print(segment)

                # Split the audio by speaker
                split_audio_by_speaker(wav_filename, transcription_segments, out_dir=out_dir)
            else:
                print("No utterances found in the transcription.")
            break

        elif transcription_result['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")