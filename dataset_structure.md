# Primary

## Observational

Files:
- `AMuCS_prequestionnaire_results.csv`: results of the prequestionnaire that was administered before the gameplay

### notes on time synchronization
LSL (labstreaming layer) was used to record synchronized data. In principle, the timestamps in all the csv files belonging in the same session are recorded from the same clock.

So far, we have only found some inconsistencies with the keyboard.csv but other inconsistencies may be present! 

### data

A directory for each session (S001-S071) and for each participant (P1-P4). The data for Session 3 participant 4 would be in `S003/P4/`

Files (per session):
- gameDictionary.csv: lookup dictionary for various enumerated data in gameInt.csv
- gamePlayerInfo.csv: information about the players in the sesssion (ex. team number)

Files (per participant):
- eyetracker.csv: data recorded with the Tobii pro nano eyetracker
- gameFlt.csv: floating point data from the gameplay (ex. position, speed, etc.). Includes a couple of derived columns which are the location of another player on the screen (for player p it is screenXp, screenYp). These values are empty if p is the same as the participant column, or the player p is not in the field of view. 
- gameInt.csv: integer data from the gameplay (ex. remaining ammo, score, etc.)
- gameMrk.csv: game markers (ex. round start/end)
- keyboard.csv: keyboard button presses
- mousebuttons.csv: mouse button presses
- mouseposition.csv: mouse position, DO NOT USE - game engine captures mouse cursor, rendering this data unusable
- obsframes.csv: frame timing information for gameplay videos
- physio.csv: bitalino data (ECG, EDA, Respiration)
- ranktrace.csv: PAGAN affect annotation (Arousal OR Valence)
- realsenseframes.csv: frame timing information for Intel RealSense recordings (face and depth)
- mat.csv: seat pressure mat data, flattened 16x16 sensor grid data

### depth videos
A directory for each session (S001-S071)

Files:
- `Px_realsense_depth.mkv`: the depth recording for participant "x", converted from the rosbag format

### gameplay videos
A directory for each session (S001-S071)

Files:
- `Px_gameplay.mp4`: the gameplay video for participant "x". Note, resolution of all videos is 1920x1080 however some participants chose to change the video settings to lower resolutions. The effect of this game resolution change was that the gameplay is on the top left corner of this 1920x1080 video. The resolutions used for each session and player are recorded in the data quality csv in the documentation.

The gameplay videos have been modified as follows:
- an algorithm was applied to filter out human speech. To this end, we used demucs https://github.com/facebookresearch/demucs .
- three videos (S063/P2, S064/P2, S067/P2) were partially blurred to anonymize the participants faces when they appeared in the recording

## Derived

### data
A directory for each session (S001-S071) and for each participant (P1-P4). The data for Session 3 participant 4 would be in `S003/P4/`

Files:
- facefeatures.csv: face features extracted from the color face video using openFace. The time of each frame was matched with the realsenseframe.csv data
- screenLuminance.csv: luminance of the screen (average over entire screen, average in central region, average in gaze position), estimated using the gameplay video cropped at the correct resolution which was used during the game.

# Secondary

## Documentation

### docs
Files:
- consent questions : list of consent questions given in the consent form prior to the experiment
- data quality: summary of the data quality of each modality for each participant, also includes gameplay video resolution and consent questions
- gamedata description: description of each column of the game data
- prequestionnaire: the questionnaire administered prior to gameplay
- dataset structure: this file

# Software

## Code

### python scripts
Files:
- read video frame (python3) : this script shows the proper way to read a frame from the depth video with the correct encodings. Also shows how to convert the values of the pixels to meters.
- sync data (python3): this script shows how to merge the data, at the end it will crop the data such that it includes only data during gameplay. Make sure to edit "basedir" (the Primary data directory of this deposit) and "outputdir" appropriately. You can specify the modality to synchronize on as the first input argument (default is 'phy' for physiological data of the bitalino sensors) and the outputdir as the second input argument to the script (default is "./synced-data/").
