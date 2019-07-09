# trash_classify
相应习大大的号召，进行垃圾分类。基于OpenCV和TensorFlow的生活垃圾图像分类识别。

## 说明
* trash_classify_demo1

基于OpenCV，对图像的二值图进行轮廓识别，并得到其边界矩形。通过此方法，大概率能够框选得到图片中的主要物体，并将框选出的方框进行裁剪为224*224的尺寸。

* trash_classify_demo2

./cnn_test.py 为此前自己摸索的卷积神经网络，训练起来准确率不佳，遂改用VGG16模型。
<br>./trash_classify_demo2/cnn_test.py 基于VGG16模型，增加bn层促使模型收敛。将训练集迭代训练约15次，训练集准确度约80%-90%，测试集准确度约60%。</br>
关于label，格式为“图片名称 类别”，由于上传大小所限，仅上传label文档，未上传数据集。

* trash_classify_demo3

一些项目进行中所编写的向程序，包括爬虫批量下载图片、调整图片尺寸、计算图片平均RGB值和生成标签文档。


### 随后将整合以上项目，并与前端衔接。
