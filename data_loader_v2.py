import os
import shutil

RAV = "./Data_loading/ravdess-emotional-speech-audio/"
dir_list = os.listdir(RAV)
dir_list.sort() #list of "Actor_1", "Actor_2" ...
os.mkdir("./big_set")
for i in dir_list: #for loops through the actor
    fname_list = os.listdir(os.path.join(RAV, i))
    fname_list.sort()
    for f in fname_list:
        shutil.move(os.path.join(RAV, i, f), "./big_set/")
