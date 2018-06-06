import pickle
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt
from librosa import display


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


if __name__ == "__main__":
    tag_class = ["normal", "extra", "murmur"]
    path_raw = "./raw_data/"
    path_data = "./data/"

    path_dic = {"normal": "/data/normal/", "murmur": "/data/murmur/", "extra": "/data/extrah/"}

    lowcut = 40.0
    highcut = 130.0

    if not os.path.exists(path_data):
        os.system("chmod +x prep_dir.sh")
        os.system("./prep_dir.sh")

    for file in os.listdir(path_raw):
        found = -1
        path_save = ""
        name_file = file.split(".")[0]
        typ = ""
        for n in tag_class:
            if file.find(n) != -1:
                found = 1
                path_save = "."+path_dic[n]
                typ = n
                break
        if found != 1:
            continue

        # if typ == "extra":
            # n_fft = 512
        # else:
        #     n_fft = 254
        n_fft = 254

        [x, fs] = librosa.load(path_raw+file, sr=None)

        if fs > 5000:
            continue

        x_f = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)

        print("sound :: ", path_save+name_file, " sample ratio:: ", fs, end=" ")

        # window FFT ~ 10 ms
        # n_fft = int((10**-2)*fs)

        # plt.rcParams['figure.figsize'] = (26.3, 1.28)  # (17, 5)
        # plt.rcParams['figure.figsize'] = (x.shape[0]/10000, 5)

        # hop length is window/4
        # magnitude of frequency <bin f at frame t>
        stft = np.abs(librosa.stft(x_f, n_fft=n_fft))

        # mel-scaled spectrogram
        mel = librosa.feature.melspectrogram(sr=fs, S=stft**2)

        kwargs = {'cmap': 'gray'}
        log_mel = librosa.amplitude_to_db(mel)

        print(" shape:: ", log_mel.shape)
        pickle.dump(log_mel, open(path_save+name_file, 'wb'))

        # adjusting
        # log_mel = log_mel[:, :2580]
        # zero padding
        # log_mel = np.pad(log_mel, ((0, 0), (25, 25)), 'constant', constant_values=0)

        # mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        # mfcc = sklearn.preprocessing.StandardScaler().fit_transform(mfcc)

        # fig = plt.figure(frameon=False)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # fig.add_axes(ax)

        # display.specshow(log_mel, sr=fs, **kwargs)
        # display.specshow(mfcc, sr=sr, x_axis='time', **kwargs)

        # fig.savefig("normal")#path_save+name_file)
        # fig.clf()
        # plt.close()
        # plt.show()

