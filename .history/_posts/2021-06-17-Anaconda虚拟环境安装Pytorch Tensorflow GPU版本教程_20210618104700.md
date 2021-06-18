# 使用Anaconda在虚拟环境中安装CUDA&CUDNN

## 前言:

使用tensorflow或者pytorch必须先安装cuda&cudnn。一般是查看当前NVDIA支持的CUDA版本，然后选择对应的CUDA&CUDNN版本号，下载CUDA&CUDNN安装包进行本地安装。但是这样会存在一个问题：如果更换tensorflow的版本，将导致CUDA版本不支持的问题。

为此，本文提出使用Anaconda虚拟环境安装CUDA&CUDNN，然后再安装pytorch或者tensorflow的GPU版本，从而做到一个pytorch或者tensorflow的GPU版本对应一个cuda&cudnn版本，减少在本机上重复安装CUDA环境的次数。

### 安装提示：

- 必须先在虚拟环境中安装CUDA，再安装pytorch/Tensorflow。否则会报错误。

- tensorflow的版本与cuda&cudnn的版本严格对应。否则会报未知错误。

## TensorflowGPU版本对应的安装教程

> 注：当前以2.3.1为例，进行安装GPU版本tensorflow

### step1：查找对应版本关系

> Tensorflow GPU版本与cuDNN&CUDA对应关系
>
> https://tensorflow.google.cn/install/source#tested_build_configurations

#### Linux/Windows系统

| 版本                  | Python 版本  | 编译器    | 构建工具     | cuDNN | CUDA |
| :-------------------- | :----------- | :-------- | :----------- | :---- | :--- |
| tensorflow-2.4.0      | 3.6-3.8      | GCC 7.3.1 | Bazel 3.1.0  | 8.0   | 11.0 |
| tensorflow-2.3.0      | 3.5-3.8      | GCC 7.3.1 | Bazel 3.1.0  | 7.6   | 10.1 |
| tensorflow-2.2.0      | 3.5-3.8      | GCC 7.3.1 | Bazel 2.0.0  | 7.6   | 10.1 |
| tensorflow-2.1.0      | 2.7、3.5-3.7 | GCC 7.3.1 | Bazel 0.27.1 | 7.6   | 10.1 |
| tensorflow-2.0.0      | 2.7、3.3-3.7 | GCC 7.3.1 | Bazel 0.26.1 | 7.4   | 10.0 |
| tensorflow_gpu-1.15.0 | 2.7、3.3-3.7 | GCC 7.3.1 | Bazel 0.26.1 | 7.4   | 10.0 |
| tensorflow_gpu-1.14.0 | 2.7、3.3-3.7 | GCC 4.8   | Bazel 0.24.1 | 7.4   | 10.0 |
| tensorflow_gpu-1.13.1 | 2.7、3.3-3.7 | GCC 4.8   | Bazel 0.19.2 | 7.4   | 10.0 |
| tensorflow_gpu-1.12.0 | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.15.0 | 7     | 9    |
| tensorflow_gpu-1.11.0 | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.15.0 | 7     | 9    |
| tensorflow_gpu-1.10.0 | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.15.0 | 7     | 9    |
| tensorflow_gpu-1.9.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.11.0 | 7     | 9    |
| tensorflow_gpu-1.8.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.10.0 | 7     | 9    |
| tensorflow_gpu-1.7.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.9.0  | 7     | 9    |
| tensorflow_gpu-1.6.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.9.0  | 7     | 9    |
| tensorflow_gpu-1.5.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.8.0  | 7     | 9    |
| tensorflow_gpu-1.4.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.5.4  | 6     | 8    |
| tensorflow_gpu-1.3.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.4.5  | 6     | 8    |
| tensorflow_gpu-1.2.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.4.5  | 5.1   | 8    |
| tensorflow_gpu-1.1.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.4.2  | 5.1   | 8    |
| tensorflow_gpu-1.0.0  | 2.7、3.3-3.6 | GCC 4.8   | Bazel 0.4.2  | 5.1   | 8    |

#### MAC OS:

| 版本                 | Python 版本  | 编译器           | 构建工具    | cuDNN | CUDA |
| :------------------- | :----------- | :--------------- | :---------- | :---- | :--- |
| tensorflow_gpu-1.1.0 | 2.7、3.3-3.6 | Xcode 中的 Clang | Bazel 0.4.2 | 5.1   | 8    |
| tensorflow_gpu-1.0.0 | 2.7、3.3-3.6 | Xcode 中的 Clang | Bazel 0.4.2 | 5.1   | 8    |

### step2：使用conda安装tensorflow2.3.1GPU版本

> 创建一个新的环境，使用conda安装cuda&cudnn。
>
> 使用conda安装cuda&cudnn，是在虚拟环境中安装的
>
> **注意:tensorflow2.3.1与cuda10.1&cudnn7.6.5版本必须严格对照。否则将会报未知错误。**

```shell
# 创建一个env_name的虚拟环境
conda create -n env_name python=3.8
# 在当前虚拟环境中安装CUDA cuDNN
# 这里的cudatoolkit就是cuda
conda install cudatoolkit=10.1 cudnn=7.6.5
# 安装tensorflow gpu
pip install tensorflow-gpu==2.3.1 -i https://pypi.mirrors.ustc.edu.cn/simple/
```

> 就三行代码，即可将tensorflow-gpu 2.3.1版本的安装成功

### step3：安装完成后，可以运行以下测试代码。测试下gpu速度。

```python
# 测试
import tensorflow as tf
tf.test.is_gpu_available()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
```

> 作者笔记本GTX 1650显卡的速度。

![image-20210607190926599](021GPU%E8%BF%90%E8%A1%8CPytorch%20Tensorflow%E7%A8%8B%E5%BA%8F%E6%AD%A5%E9%AA%A4.assets/image-20210607190926599.png)

--



# ------------------------------------------------

## Pytorch GPU版本对应的安装教程

### step1：查看本机支持的CUDA版本

查看主机支持的CUDA版本

打开NVIDIA控制面板-帮助-系统信息-组件。红圈中即CUDA支持的版本号:11.1.114

![image-20210410220838044](021GPU运行Pytorch程序步骤.assets/image-20210410220838044.png)

![image-20210410220957120](021GPU运行Pytorch程序步骤.assets/image-20210410220957120.png)

### step2：创建虚拟环境安装CUDA&CUDNN

```shell
# 创建一个env_name的虚拟环境
conda create -n env_name python=3.8
# 在当前虚拟环境中安装CUDA cuDNN
# 这里的cudatoolkit就是cuda
conda install cudatoolkit=10.1 cudnn=7.6.5
```

### step3：去Pytorch官网查看对应的安装版本

打开官网(https://pytorch.org/) - 点击install - 选择对应的环境 - 查看推荐安装的Pytorch综合包。

> 这里选择Stable(1.8.1),Windows,Pip,Python,CUDA 11.1。然后系统给出了安装命令:
>
> ```shell
> pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
> ```

<span style=color:red>**如果可以直接输入这个命令安装，则直接跳到step7**</span>。

![image-20210410223159728](021GPU运行Pytorch程序步骤.assets/image-20210410223159728.png)

![image-20210410223356732](021GPU运行Pytorch程序步骤.assets/image-20210410223356732.png)

#### 安装CUDA别的版本

打开下面这个网址，找到CUDA与pytorch对应的版本号，以及对应的安装命令。

https://pytorch.org/get-started/previous-versions/

![image-20210611092956708](021Anaconda%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85Pytorch%20Tensorflow%20GPU%E7%89%88%E6%9C%AC%E6%95%99%E7%A8%8B.assets/image-20210611092956708.png)

#### 例如pytorch1.7.1版本

![image-20210611093114166](021Anaconda%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85Pytorch%20Tensorflow%20GPU%E7%89%88%E6%9C%AC%E6%95%99%E7%A8%8B.assets/image-20210611093114166.png)



### srep4：安装Pytorch离线包

下载pytorch官网给出的命令`torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1`对应的离线包版本，进行安装。

### step5：找到离线包并下载

打开Pytorch本地安装环境包的路径地址(https://download.pytorch.org/whl/torch_stable.html) - 找到`torch==1.8.1+cu111` `torchvision==0.9.1+cu111` `torchaudio===0.8.1`三个包对应的CUDA11.1版本、本机Python版本进行安装。这里cp36、cp37分别代表python3.6、python3.7

![image-20210410223843992](021GPU运行Pytorch程序步骤.assets/image-20210410223843992.png)

> 1.用迅雷下载会快很多。
>
> 2.用迅雷下载会改变名字。出现乱码。可能需要找到原文件名，进行重命名。这里直接重命名了。

### step6：在目标环境中安装离线包

打开cmd，切换到离线包的路径，运行下面命令

```shell
pip install 离线包名字.whl
```

即可安装成功

![image-20210410210124887](021GPU运行Pytorch程序步骤.assets/image-20210410210124887.png)

### step7：安装完成后，可以运行以下命令，查看是否安装成功

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')	# 自动判断CUDA是否可以使用
# 调用方法
.to(device)		# 将数据、模型、在GPU中使用的函数/方法导入到GPU中
# 或者在初始化pytorch张量,令device=device.
torch.zeros((shape), device=device)
```

> 这里偷个懒，就不贴代码了。你自己找个可以在CPU上运行的程序，然后添加上述代码块，之后将所有的数据、模型、在GPU中使用的函数/方法使用`.to(device)`导入到GPU内存中运行测试即可。

### 备注：pytorch使用GPU运行程序

使用说明:将数据、模型、在GPU中使用的函数都导入进GPU的内存中



# -------------------------------------------------