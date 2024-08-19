from SSA import *
import config as config

i = 1
if config.yt:
    meta_data = pd.DataFrame(columns = ["filename","transcript"])
    for link in config.links:
        csv = pd.read_csv(link)
        print(csv)
        
        for index,row in csv.iterrows():
            try:
            
                url = row["Link"]
                
                start_time = row["Start_time"]
                audio_filename = 'target_speaker.wav'
                print(row["Link"],row["Start_time"])
                download_audio(url, start_time, audio_filename)
                speaker_diarization(audio_filename)
                if config.ssa_on:
                    ind,meta = sentence_splitter_voiced(f"{config.out_dir}/speaker_audios/{config.speaker_name}/speaker_A.wav",i)
                else:
                    ind,meta = segment_audio(f"{config.out_dir}/speaker_audios/{config.speaker_name}/speaker_A.wav", config.threshold, i, out_dir = f"{config.out_dir}/{config.speaker_name}/")
                # print(meta)
                # meta["total"] = meta["end"] - meta["start"]
                meta_data = pd.concat([meta_data, meta], ignore_index=True)
                # print(ind)
                i = ind
                #cleaning the music
                # rem_music(f"output/{config.speaker_name}/")
            
            except:
                print("***********************************************************************************************************************************************************************************************************************************************************")
                print(url,"not done",Exception)
                continue

        print(meta_data.total.round().value_counts())
        meta_data.to_csv(f"{config.out_dir}/{config.speaker_name}/meta_data.csv",index = False)

if config.directory:
    
    f=0
    meta_data = pd.DataFrame(columns = ["filename","transcript"])
    for link in config.dir_link:
        data_dir = (link)
        file_paths = pd.DataFrame({"path" :[os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, filename))]})
        
        file_paths = file_paths.sort_values("path",ascending=True)
        print(file_paths[0:7])
        df= pd.DataFrame(columns = ["name","duration"])
        
        for index, row in file_paths[5:6].iterrows():
            # url = a
            print(index)
            start_time = '00:00:00'
            audio_filename = row["path"]

            # download_audio(url, start_time, audio_filename)

            # speaker_diarization(audio_filename)
            # ind,meta = sentence_splitter( f"{config.out_dir}/speaker_audios/{config.speaker_name}/speaker_A.wav",i)
            if config.ssa_on:
                ind,meta = sentence_splitter_voiced(row["path"],i)
                # ind,meta = sentence_splitterf"{config.out_dir}/speaker_audios/{config.speaker_name}/speaker_A.wav",i)
            else:
                print("here")
                ind,meta = segment_audio(row["path"], config.threshold, i, out_dir = f"{config.out_dir}/{config.speaker_name}/")
                # ind,meta = segment_audio(f"{config.out_dir}/speaker_audios/{config.speaker_name}/speaker_A.wav", config.threshold, i, out_dir = f"{config.out_dir}/{config.speaker_name}/")
            print(meta)
            meta_data = pd.concat([meta_data, meta], ignore_index=True)
            print(ind)
            i = ind
            #cleaning the music
            # rem_music(f"output/{config.speaker_name}/")

        # print(meta_data.total.round().value_counts())
        meta_data.to_csv(f"{config.out_dir}/{config.speaker_name}/meta_data_audible.csv",index = False)