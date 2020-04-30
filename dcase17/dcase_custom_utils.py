import numpy as np
def load_folds(_fold_base, _mono, ws, _fold=None):
    feat_files = []
    _X_train = []
    _X_test = []
    _Y_train = []
    _Y_test = []

    subscript = '' if _fold is None else str(_fold)
    for w in ws:
        feat_file = _fold_base + subscript + '/nfft_{}_mbe_{}_fold{}.npz'.format(w, 'mon' if _mono else 'bin', _fold)
        dmp = np.load(feat_file)
        _X_train.append(dmp['arr_0'])
        _Y_train.append(dmp['arr_1'])
        _X_test.append(dmp['arr_2'])
        _Y_test.append(dmp['arr_3'])
    return _X_train, _Y_train, _X_test, _Y_test


def preprocess_data(_data, _seq_len, _ch=None):
    # split into sequences
    _data = split_in_seqs(_data, _seq_len)
    if _ch is not None:
        _data = split_multi_channels(_data, _ch)
    return _data

def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2]))
    return data

def split_multi_channels(data, num_channels):
    in_shape = data.shape
    if len(in_shape) == 3:
        hop = in_shape[2] // num_channels
        tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
        for i in range(num_channels):
            tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
    else:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", in_shape)
        exit()
    return tmp

def zero_pad_spectrograms(mel_spectrograms, window_sizes=[512, 1024, 2048, 4096, 8192, 16384]):
    spectrograms = list()
    for iw, w in enumerate(window_sizes):
        spec_size = mel_spectrograms[iw].apply(lambda v: v.shape[0])
        max_spec_size = max(spec_size.values)
        mel_spec = mel_spectrograms[iw].values
        for  i, s in enumerate(mel_spec):
            # Zero pad
            padded_s = np.pad(s, ((0, max_spec_size-s.shape[0]), (0, 0)), 'constant')
            mel_spec[i] = padded_s
            spectrograms.append(mel_spec)
    return spectrogram

def get_max_len(x_train, x_test, n_train_f, n_test_f):
    file_len_train = []
    file_len_test = []
    for i in range(len(x_train)):
        file_len_train.append(x_train[i].shape[0]//n_train_f)
        file_len_test.append(x_test[i].shape[0]//n_test_f)
    max_file_len = max(file_len_train+file_len_test)
    return max_file_len, file_len_train, file_len_test

def pad_individual_files(file, n_files, file_len, max_len):
    _X = []
    for idx, x in enumerate(file): 
        __X = []
        for s in range(n_files):
            _slice = x[ s*file_len[idx] : (s+1)*file_len[idx], : ]
            _slice = np.pad(_slice, ((0, max_len - file_len[idx]), (0, 0)), 'constant')
            __X.append(_slice)
        _X.append(np.vstack(__X))
    return _X
