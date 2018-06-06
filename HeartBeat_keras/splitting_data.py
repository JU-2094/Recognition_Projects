import os
from random import shuffle

path_data = "data/"
path_train = "train/"
path_test = "test/"
path_valid = "valid/"

if not os.path.exists(path_train):
    os.system("chmod +x prep_files.sh")
    os.system("./prep_files.sh")


data_normal = os.listdir(path_data+"normal")
data_murmur = os.listdir(path_data+"murmur")
data_extra = os.listdir(path_data+"extrah")

# 80, 10, 10
shuffle(data_normal)
shuffle(data_murmur)
shuffle(data_extra)


nor_len = len(data_normal)
mur_len = len(data_murmur)
xtr_len = len(data_extra)


def cp_to_dir(files, init, end, origin, destiny):
    for f in files[init:end]:
        origincp = origin + f
        os.system("cp "+origincp+" "+destiny)

cp_to_dir(data_normal, 0, int(nor_len/10 * 8), path_data+"normal/",
          path_train+"0/")
cp_to_dir(data_normal, int(nor_len/10 * 8)+1, int(nor_len/10 * 9), path_data+"normal/",
          path_valid+"0/")
cp_to_dir(data_normal, int(nor_len/10 * 9)+1, -1, path_data+"normal/",
          path_test+"0/")

cp_to_dir(data_murmur, 0, int(mur_len/10 * 8), path_data+"murmur/",
          path_train+"1/")
cp_to_dir(data_murmur, int(mur_len/10 * 8)+1, int(nor_len/10 * 9), path_data+"murmur/",
          path_valid+"1/")
cp_to_dir(data_murmur, int(mur_len/10 * 9)+1, -1, path_data+"murmur/",
          path_test+"1/")

cp_to_dir(data_extra, 0, int(xtr_len/10 * 8), path_data+"extrah/",
          path_train+"2/")
cp_to_dir(data_extra, int(xtr_len/10 * 8)+1, int(nor_len/10 * 9), path_data+"extrah/",
          path_valid+"2/")
cp_to_dir(data_extra, int(xtr_len/10 * 9)+1, -1, path_data+"extrah/",
          path_test+"2/")




