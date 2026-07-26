在DeepFaceLab\gui_app\文件夹里面新建文件夹models，里面存放需要的权重文件（DeepFaceLab\gui_app\models），需要的权重文件包括好几个模块；
    1、Face Parsing的权重文件（https://hf-mirror.com/jonathandinu/face-parsing/tree/main/onnx，把里面的两个权重文件下载到文件夹里面）
    2、sam2.1的权重文件（）
    3、pytorch的vgg权重文件（https://download.pytorch.org/models/vgg19-dcbb9e9d.pth）
    4、yolo的实例分割权重文件（https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt，
                               https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt，
                               https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt，
                               https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-seg.pt，
                               https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt）
insightface需要权重文件，请下载权重文件antelopev2到DeepFaceLab\insightface\models目录，官方下载地址：，将下载的文件解压，最后目录路径是：DeepFaceLab\insightface\models\antelopev2

