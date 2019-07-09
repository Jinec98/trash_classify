import cv2
import os
import numpy as np

read_path = 'D:/WorkSpace/Python/trash_classify_dataset/dataset/'


# 批量读写文件
image_list = os.listdir(read_path)
b_all = g_all = r_all = 0
image_num = 0
for image_name in image_list:
    if image_name.endswith('.jpg'):
        image_num += 1
        image_path = read_path + image_name
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        rgb = list()
        num = 0
        for a in range(h):
            for b in range(w):
                # print(img[a, b])
                rgb.append(list(img[a, b]))
                num += 1
        sumx = sumy = sumz = 0
        for i in range(num):
            [x, y, z] = rgb[i]
            sumx = sumx + x
            sumy = sumy + y
            sumz = sumz + z

        r = sumx / num
        g = sumy / num
        b = sumz / num

        r_all += r
        g_all += g
        b_all += b

        print(image_num)

r_avg = r_all / image_num
g_avg = g_all / image_num
b_avg = b_all / image_num

print(r_avg, g_avg, b_avg)