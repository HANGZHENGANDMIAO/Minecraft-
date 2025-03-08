# Minecraft-
基于YOLO算法实现的小目标物体检测，labelmg图片标注
工具
训练YOLO模型的要点
1.数据集的标注需要准确无误
2.数量足够
3.识图，看数据
4.将训练的模型调用

步骤
1-->github上拉取YOLOv5的源代码:路径
https://github.com/ultralytics/yolov5
cd到YOLO目录 pip install -r requirements.txt安装依赖
2-->labellmg工具标注数据集，一张图片一个标签，一种
类
3-->将数据集分为训练集和验证集，训练集占多数，验证集少数
4-->适当修改参数使用trian.py脚本训练模型，将训练得到的模型放到对应文件夹下
5-->使用模型完成相应功能
