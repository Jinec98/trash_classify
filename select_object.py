import cv2


def pretreatment_image(img_path, img_name, upload_path):
    path = upload_path + '\\'
    print(path)
    img = cv2.imread(img_path)
    # 灰度化
    origin_image = img.copy()
    gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

    # 二值化
    gray_image = cv2.GaussianBlur(gray_image, (21, 21), 0)  # 对灰度图进行高斯模糊的处理
    diff_image = cv2.threshold(gray_image, 160, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    # es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    # diff = cv2.dilate(diff, es, iterations=2) # 形态学膨胀

    height, width = diff_image.shape  # 获取图像尺寸
    for i in range(height):  # 图像反色，轮廓识别是依据二值图中白色的区域，因此需要将黑白调转
        for j in range(width):
            diff_image[i, j] = (255 - diff_image[i, j])

    contours, hierarchy = cv2.findContours(diff_image.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # 轮廓识别，得到所有轮廓的列表及其索引

    # 依据大小选取方框
    index = 0  # 轮廓线索引下标
    max_offset = 0  # 初始化最大值
    img_size = img.shape
    img_width = img_size[0]
    img_height = img_size[1]

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)  # 得到轮廓的边界矩形
        offset = w + h  # 求取边界矩形长宽和
        if max_offset < offset:  # 选取长宽和最大的边界矩形
            max_offset = offset
            index = c
        cv2.rectangle(origin_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制边界矩形

    (x, y, w, h) = cv2.boundingRect(index)  # 获取被选取的方框的尺寸
    e = max(w, h)  # 正方形方框的边长，取矩形方框长宽中的最大值
    if e > min(img_width, img_height):  # 如果边长大于原图像的长或宽，则取其中的最小值
        e = min(img_width, img_height)

    x = int(x - (e - w) / 2)  # 调整方框x坐标，使图像位于方框中央
    if x < 0:  # 如果x在原图像上方，将其设定为0
        x = 0
    elif x + e > img_height:  # 如果x+e（方框左下角）在原图像下方，将其设定为img_height - e
        x = img_height - e
    # 由于e小于原图像长宽中的最小值，因此将x设定为0，x+e一定小于img_height，同理将x设定为img_height - e，x+e也一定小于img_height

    # 对于y的坐标，同理
    y = int(y - (e - h) / 2)
    if y < 0:
        y = 0
    elif y + e > img_width:
        y = img_width - e

    cv2.rectangle(origin_image, (x, y), (x + e, y + e), (0, 0, 255), 2)  # 绘制边界矩形

    select_image = img[y:y + e, x:x + e]
    select_image = cv2.resize(select_image, (224, 224))

    # 保存图像
    pretrain_img_path = path + img_name + '_pretrain.jpg'
    selected_img_path = path + img_name + '_selected.jpg'
    cv2.imwrite(pretrain_img_path, select_image)
    cv2.imwrite(selected_img_path, origin_image)

    return pretrain_img_path, selected_img_path

# 单一测试图片
# pretreatment_image('test.jpg', 'test')

# print("finish!")
