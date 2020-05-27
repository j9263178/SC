import cv2
import os.path


# github找的函數，反正給圖片幫你找臉
def detect(target, filename, save_path, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    try:
        image = cv2.imread(target, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    except:
        print("not image!")
        return
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(64, 64))
    write_path = os.path.join(save_path, filename)
    for (x, y, w, h) in faces:
        #    cv2.rectangle(image, (x-50, y-50), (x + w+50, y + h+50), (0, 0, 255), 2)

        # cv2.imwrite(".png", image)

        #    cv2.imshow("AnimeFaceDetect", image)
        #    cv2.waitKey(0)
        try:
            crop = image[y - 50:y + h + 50, x - 50:x + w + 50]
            crop = cv2.resize(crop, (128, 128))
            cv2.imwrite(write_path, crop)
        except:
            print("An exception occurred")


# 給一個path，幫你把裡面的圖片全轉成np.array然後合成一個巨大資料集array ( shape=圖片數＊長*寬*3 )
def build(path):
    file_list = os.listdir(path)
    print("Found " + str(len(file_list)) + " files!")
    dataset_name = 'anime01'
    height = 64
    width = height
    channel = 3
    img_data = np.zeros((len(file_list), height, width, channel))
    # print(img_data.shape)
    for ii, file in enumerate(file_list):
        try:
            img = cv2.imread(os.path.join(path, file))
            img = cv2.resize(img, (height, width))
            img_data[ii, :, :, :] = img
        except:
            print("not image!")

    print(img_data.shape)

    # 這兩行是存資料集陣列並壓縮
    # np.save(dataset_name, img_data)
    # savez_compressed(dataset_name, img_data)
    return img_data


path = '/Users/joseph/PycharmProjects/craw/Pixiv/2019-12-1'
save_path = '/Users/joseph/PycharmProjects/gan01/dataset'

# 示範
for filename in os.listdir(path):
    target = os.path.join(path, filename)
    print(target)
    detect(target, filename, save_path)
