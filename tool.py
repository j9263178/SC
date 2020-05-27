import cv2
import numpy as np

rawcount = 1
colcount = 16
raws = []
path = "/Users/joseph/Desktop/alright/"
filename = "hihi5.png"
savepath = "/Users/joseph/Desktop/"

for j in range(1, rawcount + 1):

    rawImgs = []

    for name in range(1, colcount + 1):
        rawImgs.append(cv2.imread(path + ""+str(name) + '.png'))

    raws.append(cv2.hconcat(rawImgs))

if rawcount == 1:
    cv2.imwrite(savepath + filename, raws[0])
else:
    result = cv2.vconcat(raws)
    cv2.imwrite(savepath + filename, result)
