import cv2
import os

read_path_init = 'D:/WorkSpace/Python/trash_classify_dataset/'
write_path_init = 'D:/WorkSpace/Python/trash_classify_dataset/select/'
read_name_list = ['schoolbag']
write_name_list = ['schoolbag']

# 批量读写文件
for i in range(1):
    read_path = read_path_init + read_name_list[i] + '/'
    write_path = write_path_init + write_name_list[i] + '/'
    image_list = os.listdir(read_path)
    num = 0
    # 文件重命名
    for image_name in image_list:
        if image_name.endswith('.jpg'):
            image_path = read_path + image_name
            image_name_new = read_name_list[i] + '_' + str(num) + '.jpg'
            num += 1
            image_path_new = read_path + image_name_new
            try:
                os.rename(image_path, image_path_new)
            except Exception as e:
                continue

    for image_name in image_list:
        if image_name.endswith('.jpg'):
            image_path = read_path + image_name
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.resize(img,(224,224))
                cv2.imwrite(write_path+image_name,img)
            else:
                print(image_name)
    print('finish ' + write_name_list[i])

print('successful')