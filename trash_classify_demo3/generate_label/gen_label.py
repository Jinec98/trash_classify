import os
import cv2
import random

read_path = 'D:/WorkSpace/Python/trash_classify_dataset1/dataset/'
write_path = 'D:/WorkSpace/Python/trash_classify_dataset/dataset/'

all_file = open('D:/WorkSpace/Python/trash_classify_dataset/labels.txt', 'r')
train_file_init = open('D:/WorkSpace/Python/trash_classify_dataset/train_label1.txt', 'r')
train_file = open('D:/WorkSpace/Python/trash_classify_dataset/train_label.txt', 'w')
test_file_init = open('D:/WorkSpace/Python/trash_classify_dataset/test_label1.txt', 'r')
test_file = open('D:/WorkSpace/Python/trash_classify_dataset/test_label.txt', 'w')


# 整合数据集，生成“名称 类别”文档
# dir_list = os.listdir(read_path)
# class_num = 0
# for dir_name in dir_list:
#     print(dir_name)
#     class_num += 1
#     num = 0
#     dir_path = read_path + dir_name +'/'
#     image_list = os.listdir(dir_path)
#     for image_name_old in image_list:
#         num += 1
#         image_name_new = dir_name + str(num) + '.jpg'
#         old_path = dir_path + image_name_old
#         new_path = write_path + image_name_new
#         # img = cv2.imread(old_path)
#         # cv2.imwrite(new_path,img)
#         train_file.write(image_name_new + ' ' + str(class_num) + '\n')
#         # print(image_name_new + ' ' + str(class_num))


# 将数据集分成训练集和测试集
# num = 0
# while True:
#     lines = all_file.readline() # 整行读取数据
#     if not lines:
#       break
#
#     if num % 10 == 0:
#         test_file.write(lines)
#     else:
#         train_file.write(lines)
#     num +=1


# 打乱文档顺序
def read_file(file):
    data = []
    for line in file:
        line = line.strip('\n')  # 删除每一行的\n
        data.append(line)
    print('len ( data ) = ', len(data))
    return data


def write_file(data, file):
    for index in range(len(data)):
        str_data = data[index]
        file.write(str_data + '\n')


train_data = read_file(train_file_init)
random.shuffle(train_data)
write_file(train_data, train_file)

test_data = read_file(test_file_init)
random.shuffle(test_data)
write_file(test_data, test_file)

print('success!')
