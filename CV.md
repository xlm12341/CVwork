二值图像 灰度图像 彩色图像



IOU交并比 越高难度越大



 基于模板的人脸匹配？





adaboost:

多个弱分类器级联得到强分类器



后来面部识别难度加大，人脸形态，环境不同

非约束条件下的人脸检测



基于深度学习的方法

卷积神经网络，但直接使用滑动窗口加神经网络会导致计算量过大







NMS 非极大值抑制，去重 比如多个框识别同一个人脸



预训练：先初步渐小到结果域的距离



神经网络：层数越多越抽象





人脸检测vs人脸识别

人脸识别：

**基于特征匹配的方法**

数据预处理

协方差矩阵

求特征值，特征向量

依据取大的一部分特征值，基 子空间

原始矩阵投影到子空间，降维矩阵





平均脸 为什么平均脸更好看



**基于深度学习的方法**

CNN

提取特征向量，替代人工设计的特征

DeepFace CVPR 2014



人脸对齐->6个卷积层->2个全连接层





**人脸验证**



人脸关键点检测/定位/人脸对齐





关键点粗定位

关键点微调







## CV_WORK1

dlib的facial_landmarks 检测：

- 0-16 下颚
- 17-21 左眉毛
- 22-26 右眉毛
- 27-30 鼻子
- 31-35 鼻孔
- 36-41 左眼
- 42-47右眼
- 48-60 嘴唇
- 61-67 嘴巴中间