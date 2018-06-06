import os
import pickle
import numpy as np
import scipy.misc


def normalize(path):
    for d in os.listdir(path):
        path_t = path + d + "/"
        for file in os.listdir(path_t):
            path_f = path_t + file
            data = pickle.load(open(path_f, 'rb'))
            print(data.shape)


def adjust(path, size=512):
    for d in os.listdir(path):
        path_t = path + d + "/"
        for file in os.listdir(path_t):
            path_f = path_t + file
            data = pickle.load(open(path_f, 'rb'))
            # repetition
            if data.shape[1] <= size:
                while data.shape[1] < size:
                    data = np.concatenate([data, data], axis=1)
                    data = data[:, 0:size]
                os.remove(path_f)

                # pickle.dump(data, open(path_f, 'wb'))
                scipy.misc.imsave(path_f+".jpg", data)
            else:
                i = 0
                c = 0
                # print("saving.. ", path_f + "_" + str(c) + ".png")
                # scipy.misc.imsave(path_f + ".png", data)

                while i < data.shape[1]:
                    win = data[:, i:i+size]

                    if size > win.shape[1] > int(size/2):
                        win = np.concatenate([win, data[0:size]], axis=1)
                        win = win[:, 0:size]
                    elif win.shape[1] < size:
                        break
                    print("saving.. ", path_f+"_"+str(c)+".jpg")
                    scipy.misc.imsave(path_f+"_"+str(c)+".jpg", win)
                #     # pickle.dump(win, open(path_f + "_" + str(c), 'wb'))
                    i += size + 1
                    c = c+1
                os.remove(path_f)

path_train = "./train/"
path_valid = "./valid/"
path_test = "./test/"

adjust(path_train)
adjust(path_valid)
adjust(path_test)
