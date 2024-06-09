from SSA import *
import config as config

if config.yt:
    i = 1
    for link in config.links:
        csv = pd.read_csv(link)
        print(csv)
        
        for index,row in csv.iterrows():
            url = row["Link"]
            start_time = row["Start_time"]
            audio_filename = 'target_speaker.wav'
            print(row["Link"],row["Start_time"])
            download_audio(url, start_time, audio_filename)
            speaker_diarization(audio_filename)
            ind,meta = sentence_splitter(f"speaker_audios/{config.speaker_name}/speaker_A.wav",i)
            print(meta)
            meta["total"] = meta["end"] - meta["start"]
            meta_data = pd.concat([meta_data, meta], ignore_index=True)
            print(ind)
            i = ind
            #cleaning the music
            rem_music(f"output/{config.speaker_name}/")

        print(meta_data.total.round().value_counts())
        meta_data.to_csv(f"output/{config.speaker_name}/meta_data.csv",index = False)

else:
    i = 0
    f=0
    meta_data = pd.DataFrame(columns = ["filename","text","start","end","total"])
    for link in config.dir_link:
        data_dir = (link)
        file_paths = pd.DataFrame({"path" :[os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, filename))]})
        
        file_paths = file_paths.sort_values("path",ascending=True)
        print(file_paths)
        df= pd.DataFrame(columns = ["name","duration"])
        
        for index, row in file_paths.iterrows():
            
            # url = a
            start_time = '00:00:00'
            audio_filename = row["path"]

            # download_audio(url, start_time, audio_filename)

            speaker_diarization(audio_filename)
            ind,meta = sentence_splitter(f"speaker_audios/{config.speaker_name}/speaker_A.wav",i)
            print(meta)
            meta["total"] = meta["end"] - meta["start"]
            meta_data = pd.concat([meta_data, meta], ignore_index=True)
            print(ind)
            i = ind
            #cleaning the music
            rem_music(f"output/{config.speaker_name}/")

        print(meta_data.total.round().value_counts())
        meta_data.to_csv(f"output/{config.speaker_name}/meta_data.csv",index = False)