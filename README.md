# Semi-supervised classification by Graph Convolutional Network with Keras 使用Keras实现的图卷积神经网络

**Clike Here for** [English Version](#English_Version)

**Attention！由于Python第三方库更新频繁，请确保使用的库版本与requirement.txt中的保持一致，否则容易出现不兼容的情况**

使用Keras实现，用于半监督节点分类的图卷积神经网络。相比如作者提供的源代码，重写了部分主函数和功能块，使其比源代码更加简洁同时算法的性能与原论文中描述结果保持一致。感谢大佬的开源代码：

1. TensorFlow : <https://github.com/tkipf/gcn>
2. Keras : <https://github.com/tkipf/keras-gcn>

如果您想了解更多关于图卷积神经网络 (GCN) 的原理，请参考作者的原论文，博客以及知乎上的回答：

1. Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)

2. Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

3. [知乎：如何理解 Graph Convolutional Network（GCN）？](https://www.zhihu.com/question/54504471/answer/332657604)


## 依赖库和安装
在运行此Python代码时，需要安装文件'requirements.txt'中的Python库。安装很简单，只需下面一条命令即可：

    $ pip install -r requirements.txt

## 用法和性能
默认使用的数据集是Cora网络，关于数据集的详细介绍能在文件夹data中找到。在该项目的目录中，执行下面的一条命令就行能运行：

    $ python train.py

该代码的性能与原论文中结果保持一致，代码最后输出如下:
```
2708/2708 [==============================] - 0s 7us/step
Test Done.
Test loss: 1.0794732570648193
Test accuracy: 0.8139998912811279

[Done] exited with code=0 in 25.405 seconds
```
推荐使用tensorflow-GPU的版本，我的GPU是NVIDIA GTX 1060。如果你使用更好的GPU（比如 TITAN RTX ），能跑的更快！:heart_eyes:






# English_Version
##  Semi-supervised classification by Graph Convolutional Network with Keras
Keras-based implementation of graph convolutional networks **(GCN)** for semi-supervised classification. Rewrite a part of main function and some utils which is more simple compared the author's implementation. Thanks for his open source code at the following links :

1. TensorFlow : <https://github.com/tkipf/gcn>
2. Keras : <https://github.com/tkipf/keras-gcn>

For a more detail explanation of the GCN, have a look at the relevent paper and blog post by the orignal author :

1. Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)

2. Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)


## Dependencies and Installation

Before to execute this algorithm writed in Python, it is necessary to install these required packages shown in the file named ' requirements.txt '. You can also install all the required packages by just using one command :

    $ pip install -r requirements.txt

## Usage and Performance

The default dataset is Cora Network and the detail description can be found in the file data. Just execute the following command from the project home directory :

    $ python train.py

The performance consists with the benchmark described in the paper. The partial of output are following :

```
2708/2708 [==============================] - 0s 7us/step
Test Done.
Test loss: 1.0794732570648193
Test accuracy: 0.8139998912811279

[Done] exited with code=0 in 25.405 seconds
```

Recommand to use the tensorflow-GPU as backend and My GPU is NVIDIA GTX 1060. You can run it faster with the better GPU.

## Cite

Please cite the paper if you use this code in your own work:

```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```



