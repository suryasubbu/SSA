#Sentence Splitter Algorithm (SSA)
The Sentence Splitter Algorithm (SSA) is designed to split audio into required duration chunks. It leverages several modules, including YouTube Downloader, Speaker Diarization, Automatic Speech Recognition, Timestamp Prediction, and Noise Checker.

###Modules
YouTube Downloader: Downloads audio from YouTube videos.
Speaker Diarization: Identifies and separates different speakers in the audio.
Automatic Speech Recognition: Converts speech to text.
Timestamp Prediction: Predicts timestamps for splitting the audio.
Noise Checker: Checks and handles noise in the audio.
Repository Structure
SSA.py: Contains all the functions.
config.py: Holds all the configuration variables.
Main.py: Executes the functions.
Steps for Initiation
Clone the Repository

  ```bash
    git clone <repository-url>
    cd <repository-directory>

  ```bash
    pip install -r requirements.txt


1)Change the speaker name in config.speaker_name.
2)Set the duration threshold in config.threshold.
3)Provide the directory path in config.links inside the array.
4)Provide YouTube Links (if applicable)

If you are using a YouTube link, create a CSV file separated by commas with the following format:
link,start_time
Here, start_time refers to the target speaker's speech starting time in the YouTube video. Make sure to change the speaker name accordingly.

Run the Main Script

Execute the main script:

```bash
python Main.py
