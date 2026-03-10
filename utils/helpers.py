import librosa
import numpy as np


def extract_mfcc_features(audio_path, n_mfcc=40):

    try:
        audio, sample_rate = librosa.load(audio_path, sr=22050)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc
        )

        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception as e:
        print("Audio processing error:", e)
        return None