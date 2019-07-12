# trash_classify
响应习大大的号召，进行垃圾分类。基于OpenCV和TensorFlow的生活垃圾图像分类识别。

## 项目demo说明
* trash_classify_demo1

基于OpenCV对图像的二值图进行轮廓识别，并得到其边界矩形。通过此方法，大概率能够框选得到图片中的主要物体，并基于框选出的方框对图像进行裁剪为224*224的尺寸。

* trash_classify_demo2

./cnn_test.py 为此前自己摸索的卷积神经网络，训练起来准确率不佳，遂改用VGG16模型。
<br>./trash_classify_demo2/cnn_test.py 基于VGG16模型，增加bn层促使模型收敛。将训练集迭代训练约15次，训练集准确度约80%-90%，测试集准确度约60%。</br>
关于label，格式为“图片名称 类别”，由于上传大小所限，仅上传label文档，未上传数据集。

* trash_classify_demo3

一些项目进行中所编写的小程序，包括爬虫批量下载图片、调整图片尺寸、计算图片平均RGB值和生成标签文档。

* trash_classify_demo4

程序的web前端界面。
包括图像上传、识别功能，垃圾分了科普功能，显示模型数据功能。
图像上传识别后，能给出模型预测的结果、属于的垃圾种类及其预测概率，同时对该类垃圾进行科普。


## 项目汇总

* trash_classify

前后端结合的完整项目。
<br>通过拍照上传图像，可将图像中的物品识别为干垃圾、湿垃圾、有害垃圾和可回收垃圾四类。</br>
<br>基于OpenCV轮廓识别在图像中框选出主要的物体，基于VGG16模型训练神经网络，将框选得到的图片进行预测，并将结果返回前端显示。</br>
目前模型训练集准确度83.8%，测试集准确度67.5%，仍有待提高。。

![ZWz9Z8.png](https://s2.ax1x.com/2019/07/12/ZWz9Z8.png)

[![ZWzPIg.png](https://s2.ax1x.com/2019/07/12/ZWzPIg.png)](https://imgchr.com/i/ZWzPIg)

### 完结撒花！~
