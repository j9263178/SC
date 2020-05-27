import numpy as np
import matplotlib.pyplot as plt

def solve_puzzle(img):
    ret = img.copy()
    ### START YOUR CODE HERE ###
    import cv2

    map = [-1, -1, -1, -1, -1, -1, -1, -1, -1]

    class js:
        def __init__(self, con):
            self.con = con
            self.up = []
            self.down = []
            self.left = []
            self.right = []
            for i in range(0, 120):
                self.up.append(con[0][i])
            for i in range(0, 120):
                self.down.append(con[-1][i])
            for i in range(0, 120):
                self.left.append(con[i][0])
            for i in range(0, 120):
                self.right.append(con[i][-1])
            self.near = [0, 0, 0, 0]

    def Take_Jigsaws_From_Img(img):
        h, w = 360, 360
        st = h // 3
        jss = []
        for i in range(0, 3):
            for j in range(0, 3):
                crop_img = img[st * i:st * i + st, st * j:st * j + st]
                jss.append(crop_img)
        res = []
        for j in jss:
            res.append(js(j))
        return res

    def score(a, b):
        sum = []
        for i in range(0, 120):
            pixel = []
            for j in range(0, 3):
                pixel.append(abs(int(a[i][j]) - int(b[i][j])))
            sum.append(pixel)
        rsum = 0
        gsum = 0
        bsum = 0
        for i in range(0, 120):
            rsum += sum[i][0]
            gsum += sum[i][1]
            bsum += sum[i][2]
        return (rsum / 120 + gsum / 120 + bsum / 120) / 3

    def evaluate_Center_Score(cur):

        upmin, downmin, leftmin, rightmin = 100, 100, 100, 100

        for j in range(0, len(jss)):
            if jss[j] != cur:
                a, b, c, d = score(cur.up, jss[j].down), score(cur.down, jss[j].up), \
                             score(cur.left, jss[j].right), score(cur.right, jss[j].left)

                if a < upmin:
                    upmin = a
                    cur.near[0] = jss[j]
                if b < downmin:
                    downmin = b
                    cur.near[1] = jss[j]
                if c < leftmin:
                    leftmin = c
                    cur.near[2] = jss[j]
                if d < rightmin:
                    rightmin = d
                    cur.near[3] = jss[j]

        return (upmin + downmin + leftmin + rightmin) / 4

    def Find_Center():
        center_score = 1000
        center = 0
        for js in jss:
            score = evaluate_Center_Score(js)
            if score < center_score and not checkIfDuplicates(js.near):
                center = js
                center_score = score
        return center

    def Fill_Cross():
        center = Find_Center()
        map[4], map[1], map[7], map[3], map[5] = center, center.near[0], center.near[1], center.near[2], center.near[3]
        jss.remove(center)
        jss.remove(center.near[0])
        jss.remove(center.near[1])
        jss.remove(center.near[2])
        jss.remove(center.near[3])

    def Fill_Corner():
        ulmin, urmin, dlmin, drmin = 1000, 1000, 1000, 1000
        for js in jss:
            ulscore = (score(js.right, map[1].left) + score(js.down, map[3].up)) / 2
            urscore = (score(js.left, map[1].right) + score(js.down, map[5].up)) / 2
            dlscore = (score(js.up, map[3].down) + score(js.right, map[7].left)) / 2
            drscore = (score(js.left, map[7].right) + score(js.up, map[5].down)) / 2

            if ulscore < ulmin:
                ulmin = ulscore
                map[0] = js
            if urscore < urmin:
                urmin = urscore
                map[2] = js
            if dlscore < dlmin:
                dlmin = dlscore
                map[6] = js
            if drscore < drmin:
                drmin = drscore
                map[8] = js

    def checkIfDuplicates(listOfElems):
        if len(listOfElems) == len(set(listOfElems)):
            return False
        else:
            return True

    def concat():
        raws = []
        for i in range(0, 3):
            raw = []
            for j in range(0, 3):
                raw.append(map[3 * i + j].con)
            raws.append(cv2.hconcat(raw))

        return cv2.vconcat(raws)

    jss = Take_Jigsaws_From_Img(img)
    Fill_Cross()
    Fill_Corner()
    ret = concat()
    #### END YOUR CODE HERE ####
    return ret

if __name__ == '__main__':
    data = np.load('jigsaw_data.npy')
    
    idx = np.random.randint(10)
    ret = solve_puzzle(data[idx])
    
    fig = plt.figure(figsize=(6, 6), dpi=80)
    plt.imshow(ret)
    plt.show()