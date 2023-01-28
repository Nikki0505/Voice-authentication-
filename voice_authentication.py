import os
from pyannote.audio.features import YaafeMFCC
from pyannote.database import get_protocol, FileFinder
from pyannote.audio.train.speaker import GMM# Define the location of the audio files for training and testing
database = {'speaker1': r'C:\Users\singh\Downloads\Voice-Biometrics-main\mantsha\man2.wav',
           'speaker2': r'C:\Users\singh\Downloads\Voice-Biometrics-main\aniket\ani1.wav'}
protocol = get_protocol('SpeakerDiarization', **database)# Initialize the feature extractor
mfcc = YaafeMFCC(e=False, De=False, DDe=False, coefs=20)# Train the GMM model on the audio files of speaker1
speaker1_gmm = GMM(mfcc, n_components=8, covariance_type='diag')
speaker1_gmm.fit(protocol, 'speaker1')# Train the GMM model on the audio files of speaker2
speaker2_gmm = GMM(mfcc, n_components=8, covariance_type='diag')
speaker2_gmm.fit(protocol, 'speaker2')# Get the audio file for authentication
audio_file = r'C:\Users\singh\Downloads\Voice-Biometrics-main\references'# Extract the MFCC features from the authentication audio file
mfcc_features = mfcc(audio_file)# Compare the extracted features to the GMM models of each speaker
score1 = speaker1_gmm.score(mfcc_features)
score2 = speaker2_gmm.score(mfcc_features)# Determine which speaker the authentication audio file belongs to
if score1 > score2:
    print("The authentication audio file belongs to speaker1")
else:
    print("The authentication audio file belongs to speaker2")