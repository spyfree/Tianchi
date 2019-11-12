# 天池flink 垃圾分类代码

代码分为三部分，

## 线下预训练代码
在pre_train中pretrain.py
代码说明请见注释

生成的h5下载链接为:  https://pan.baidu.com/s/16Q5QcIO9IV6xwdM9XsQJwg 提取码: 2j6i 

## 提交天池python代码
在python_package中，包含model.py, 为最高分记录代码
其他代码为之前版本以及相关测试代码

model.py会预先加载我们线下使用自己收集的图片训练的，基于nasnet large模型预训练
好的模型。(需要下载,见之前链接)

继续放在天池环境训练，之后保存为saved_model格式,供zoo api 使用
相关调参记录见子文件夹中README

## flink zoo相关代码
在garbage_image文件夹中, 使用mvn clean package 打包成garbage_image-1.0-SNAPSHOT.jar包,同样已经上传到百度云中，下载地址:
链接: https://pan.baidu.com/s/1If8TkWAIIuqSZKjNDN1bvA 提取码: nwih 


代码说明:
UpdateDebugFlatMap.java 加载模型，调整模型的mean和scale值，以及设置
Reverse input 为false,以及输入size等参数。

ProcessingFlatMap.java 进行图片预处理，读入rgb格式的图片，resize为输入为331,331大小，并调整输入格式到CHW。

RunMain.java 为flink主入口程序,进行预测
