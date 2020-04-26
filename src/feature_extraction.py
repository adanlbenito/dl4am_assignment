import librosa
import numpy as np
import pandas as pd
import math

def compute_mel_spectrograms(data, window_sizes=[512, 1024, 2048, 4096, 8192, 16384], 
    hop_size=512, new_fs=16000, n_mels=128):
    mel_df = data.filter(['name'], axis=1)

    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes]

    for w in window_sizes:
        mel_df[str(w)] = None

    for idx, row in data.iterrows():
        samples = row['samples']
        fs = row['fs']
        re_samples = librosa.resample(samples, fs, 16000)
        for w in window_sizes:
            mel_spec = librosa.feature.melspectrogram(y=re_samples, sr=new_fs,
                                                    n_fft=w, 
                                                    hop_length=hop_size,
                                                    n_mels=n_mels
                                                    ).T
            mel_spec = np.log(1+10000*mel_spec)
            mel_df[str(w)][idx] = mel_spec

    mel_df.reset_index(drop=True)
    return mel_df

def zero_pad_spectrograms(mel_spectrograms, window_sizes=[512, 1024, 2048, 4096, 8192, 16384]):
    spectrograms = list()

    for w in window_sizes:
        spec_size = mel_spectrograms[str(w)].apply(lambda v: v.shape[0])
        max_spec_size = max(spec_size.values)

        mel_spec = mel_spectrograms[str(w)].values
        for i, s in enumerate(mel_spec):
            # Zero pad
            padded_s = np.pad(s, ((0, max_spec_size-s.shape[0]), (0, 0)), 'constant')
            mel_spec[i] = padded_s

        spectrograms.append(pd.DataFrame(mel_spec))
    return spectrograms
