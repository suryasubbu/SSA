# Sentence Splitter Algorithm (SSA)
  The Sentence Splitter Algorithm (SSA) is designed to split audio into required duration chunks. It leverages several modules, including YouTube Downloader, Speaker Diarization, Automatic Speech Recognition, Timestamp Prediction, and Noise Checker.

### Modules
  1) YouTube Downloader: Downloads audio from YouTube videos.
  2) Speaker Diarization: Identifies and separates different speakers in the audio.
  3) Automatic Speech Recognition: Converts speech to text.
  4) Timestamp Prediction: Predicts timestamps for splitting the audio.
  5) Noise Checker: Checks and handles noise in the audio.
### Repository Structure
  1) SSA.py: Contains all the functions.
  2) config.py: Holds all the configuration variables.
  3) Main.py: Executes the functions.
### Steps for Initiation
  Clone the Repository and install the requirements


1) Change the speaker name in config.speaker_name.
2) Set the duration threshold in config.threshold.
3) Provide the directory path in config.links inside the array.
4) Provide YouTube Links (if applicable)

  If you are using a YouTube link, create a CSV file separated by commas with the following format:
  **link,start_time**
  Here, start_time refers to the target speaker's speech starting time in the YouTube video. Make sure to change the speaker name accordingly.

Run the Main Script

Execute the main script:

1) ```bash
   python Main.py


