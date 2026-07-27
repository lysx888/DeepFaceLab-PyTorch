# **DeepFaceLab-PyTorch说明**
因为DeepFaceLab是基于TensorFlow框架，所以我用Pytorch重新开发了一个换脸项目，实现DFL的所有功能。
# **使用说明**
1. 首先下载本项目的所有文件，解压到你想安装的盘。
2. 把需要的权重文件下载到相应的目录（[百度网盘](https://pan.baidu.com/s/5jI2r3jELVIgKRDEfOurnaw "点击进入百度网盘下载")）。
3. 多GPU暂不支持，预留了部分代码，因为没有多个GPU，所以暂时放弃，建议有能力的人开发，调试，支持协助开发。
# **目前进度**
已经完成了视频分割，人脸提取，遮罩相关，目前正在调试模型训练，特别说明：为了限制pytorch导致的cpu的满负荷，特意临时添加了SET OMP_NUM_THREADS=4和SET MKL_NUM_THREADS=4
# **项目展示**
![DeepFaceLab-PyTorch](https://raw.githubusercontent.com/lysx888/DeepFaceLab-PyTorch/refs/heads/main/docs/1.png)
![DeepFaceLab-PyTorch](https://raw.githubusercontent.com/lysx888/DeepFaceLab-PyTorch/refs/heads/main/docs/2.png)
![DeepFaceLab-PyTorch](https://raw.githubusercontent.com/lysx888/DeepFaceLab-PyTorch/refs/heads/main/docs/3.png)
![DeepFaceLab-PyTorch](https://raw.githubusercontent.com/lysx888/DeepFaceLab-PyTorch/refs/heads/main/docs/4.png)
![DeepFaceLab-PyTorch](https://raw.githubusercontent.com/lysx888/DeepFaceLab-PyTorch/refs/heads/main/docs/5.png)
![DeepFaceLab-PyTorch](https://raw.githubusercontent.com/lysx888/DeepFaceLab-PyTorch/refs/heads/main/docs/6.png)
![DeepFaceLab-PyTorch](https://raw.githubusercontent.com/lysx888/DeepFaceLab-PyTorch/refs/heads/main/docs/7.png)
![DeepFaceLab-PyTorch](https://raw.githubusercontent.com/lysx888/DeepFaceLab-PyTorch/refs/heads/main/docs/8.png)
