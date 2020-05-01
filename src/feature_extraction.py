import librosa
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

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
        re_samples = librosa.resample(samples, fs, new_fs)
        for w in window_sizes:
            mel_spec = librosa.feature.melspectrogram(y=re_samples, sr=new_fs,
                                                    n_fft=w, 
                                                    hop_length=hop_size,
                                                    n_mels=n_mels,
                                                    power=1
                                                    ).T
            #mel_spec = np.log(1+10000*mel_spec)
            mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
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

def standardize_spectrogram(spectrogram, scaler=None):
    spec = np.apply_along_axis(lambda x: x[0], 1, spectrogram.values)
    s_shape =  spec.shape
    spec = np.reshape(spec,(-1, s_shape[2]))
    if scaler is None:
        scaler = StandardScaler().fit(spec)
    s_scaled = scaler.transform(spec)
    s_scaled = np.reshape(s_scaled, s_shape)
    return s_scaled, scaler


def prepare_input_dimensions(spectrogram):
    spec = np.expand_dims(spectrogram, axis=0)
    s_shape = spec.shape
    spec = np.reshape(spec,(s_shape[1], s_shape[3], s_shape[2], s_shape[0]))
    return spec
