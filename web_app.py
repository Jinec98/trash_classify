from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
import os

from vgg16_model import trash_classify

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])  # 设置允许的文件格式


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.send_file_max_age_default = timedelta(seconds=1)  # 设置静态文件缓存过期时间


@app.route('/', methods=['GET'])
def index():
    return render_template("Trash_selected.html")


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        base_path = os.path.dirname(os.path.realpath(__file__))  # 获取脚本路径
        upload_path = os.path.join(base_path, 'static\images')  # 上传文件目录
        if not os.path.exists(upload_path):  # 判断文件夹是否存在
            os.makedirs(upload_path)

        filedata = request.files.get('fileField')  # 获取前端对象
        if not (filedata and allowed_file(filedata.filename)):  # 检查文件类型
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
        file_path = os.path.join(upload_path, secure_filename(filedata.filename))  # 指定保存文件夹的路径
        file_name = "img.jpg"

        try:
            # 上传文件
            filedata.save(file_path)
            print(file_path, file_name, upload_path)
            #-------------------后端数据在这里!------------------
            listtest = trash_classify(file_path, file_name, upload_path)
            datalist = sort(listtest)
            return render_template('Trash_selected_ok.html',  possible=datalist[0], name=datalist[1],
                                   catalog=datalist[2], same_catalog=datalist[3], fpf=upload_path)
        except IOError:
            return jsonify({'code': -1, 'msg': '上传失败，请重试！'})
    else:
        return render_template('Trash_selected.html')

@app.route('/href')
def href():
    return render_template('trash.html')



def sort(datalist):
    num = int(datalist[0]) + 1
    possible = round(datalist[1], 3)

    if num == 1:
        name = '苹果'
        catalog = '湿垃圾'
    elif num == 2:
        name = '香蕉'
        catalog = '湿垃圾'
    elif num == 3:
        name = '电池'
        catalog = '有害垃圾'
    elif num == 4:
        name = '豆类'
        catalog = '湿垃圾'
    elif num == 5:
        name = '面包'
        catalog = '湿垃圾'
    elif num == 6:
        name = '灯泡'
        catalog = '有害垃圾'
    elif num == 7:
        name = '蛋糕'
        catalog = '湿垃圾'
    elif num == 8:
        name = '易拉罐'
        catalog = '可回收垃圾'
    elif num == 9:
        name = '衣服'
        catalog = '可回收垃圾'
    elif num == 10:
        name = '干燥剂'
        catalog = '干垃圾'
    elif num == 11:
        name = '包子'
        catalog = '湿垃圾'
    elif num == 12:
        name = '眼镜'
        catalog = '干垃圾'
    elif num == 13:
        name = '薯条'
        catalog = '湿垃圾'
    elif num == 14:
        name = '玻璃瓶'
        catalog = '可回收垃圾'
    elif num == 15:
        name = '饺子'
        catalog = '湿垃圾'
    elif num == 16:
        name = '汉堡包'
        catalog = '湿垃圾'
    elif num == 17:
        name = '猕猴桃'
        catalog = '湿垃圾'
    elif num == 18:
        name = '日光灯'
        catalog = '有害垃圾'
    elif num == 19:
        name = '柠檬'
        catalog = '湿垃圾'
    elif num == 20:
        name = '打火机'
        catalog = '有害垃圾'
    elif num == 21:
        name = '餐盒'
        catalog = '干垃圾'
    elif num == 22:
        name = '肉类'
        catalog = '湿垃圾'
    elif num == 23:
        name = '面条'
        catalog = '湿垃圾'
    elif num == 24:
        name = '橘子'
        catalog = '湿垃圾'
    elif num == 25:
        name = '油漆桶'
        catalog = '有害垃圾'
    elif num == 26:
        name = '纸'
        catalog = '可回收垃圾'
    elif num == 27:
        name = '纸箱'
        catalog = '可回收垃圾'
    elif num == 28:
        name = '纸杯'
        catalog = '干垃圾'
    elif num == 29:
        name = '梨'
        catalog = '湿垃圾'
    elif num == 30:
        name = '笔'
        catalog = '干垃圾'
    elif num == 31:
        name = '杀虫剂'
        catalog = '有害垃圾'
    elif num == 32:
        name = '药片'
        catalog = '有害垃圾'
    elif num == 33:
        name = '火龙果'
        catalog = '湿垃圾'
    elif num == 34:
        name = '披萨'
        catalog = '湿垃圾'
    elif num == 35:
        name = '塑料袋'
        catalog = '干垃圾'
    elif num == 36:
        name = '塑料瓶'
        catalog = '可回收垃圾'
    elif num == 37:
        name = '粥'
        catalog = '湿垃圾'
    elif num == 38:
        name = '土豆'
        catalog = '湿垃圾'
    elif num == 39:
        name = '米饭'
        catalog = '湿垃圾'
    elif num == 40:
        name = '书包'
        catalog = '可回收垃圾'
    elif num == 41:
        name = '鞋'
        catalog = '干垃圾'
    elif num == 42:
        name = '汤'
        catalog = '湿垃圾'
    elif num == 43:
        name = '草莓'
        catalog = '湿垃圾'
    elif num == 44:
        name = '体温计'
        catalog = '有害垃圾'
    elif num == 45:
        name = '卫生纸'
        catalog = '干垃圾'
    elif num == 46:
        name = '牙刷'
        catalog = '干垃圾'
    elif num == 47:
        name = '毛巾'
        catalog = '干垃圾'
    elif num == 48:
        name = '雨伞'
        catalog = '干垃圾'
    elif num == 49:
        name = '水杯'
        catalog = '可回收垃圾'
    elif num == 50:
        name = '编织袋'
        catalog = '干垃圾'

    if catalog == '干垃圾':
        same_catalog = '''干垃圾是指除有害垃圾、可回收物、湿垃圾以外的其他生活废弃物。
                       投放要求:
                       ● 尽量沥干水分
                       ● 难以辨识类别的生活垃圾投入干垃圾容器内
                       如1号电池(无汞)、3D眼镜、504胶水、胶水废包装可回收物等'''
    elif catalog == '湿垃圾':
        same_catalog = '''湿垃圾是指易腐的生物质废弃物。包括剩菜剩饭、瓜皮果核、花卉绿植、肉类碎骨、过期食品、餐厨垃圾等。
                       投放要求:
                       ● 纯流质的食物垃圾，如牛奶等，应直接倒进下水
                       ● 有包装物的湿垃圾应将包装物取出后分类投放，包装物请投放到对应的可回收物容器或干垃圾容器
                       如COCO里的青稞等'''
    elif catalog == '有害垃圾':
        same_catalog = '''有害垃圾是指对人体健康或者自然环境造成直接或者潜在危害的零星废弃物，单位集中产生的除外。主要包括废电池、废灯管、废药品、废油漆桶等。
                       投放要求
                       ● 充电电池、纽扣电池、蓄电池投放时请注意轻放
                       ● 油漆桶、杀虫剂如有残留请密闭后投放
                       ● 荧光灯、节能灯易破损请连带包装或包裹后轻放
                       ● 废药品及其包装连带包装一并投放'''
    elif catalog == '可回收垃圾':
        same_catalog = '''可回收物是指适宜回收和可循环再利用的废弃物。主要包括废玻璃、废金属、废塑料、废纸张、废织物等。
                       投放要求:
                       ● 轻投轻放,清洁干燥，避免污染
                       ● 废纸尽量平整
                       ● 立体包装请清空内容物，清洁后压扁投放
                       ● 有尖锐边角的，应包裹后投放'''

    list = [possible, name, catalog, same_catalog]
    return list

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8987, debug=True, use_reloader=False)