import mxnet as mx
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
n = 40
#  输入形状: batch x channel x height x weight
x = nd.random_uniform(shape=(1, 3, n, n))
y = MultiBoxPrior(x, sizes=[.5, .25, .1], ratios=[1, 2, .5])
# 取位于 (20,20) 像素点的第一个预设框
# 格式为 (x_min, y_min, x_max, y_max)
boxes = y.reshape((n, n, -1, 4))
print('The first anchor box at row 21, column 21:', boxes[20, 20, 0, :])

import matplotlib.pyplot as plt
def box_to_rect(box,color,linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box=box.asnumpy()
    return plt.Rectangle(
        (box[0],box[1]),(box[2]-box[0]),(box[3]-box[1]),
        fill=False,edgecolor=color,linewidth=linewidth)
colors=['blue','green','red','black','magenta']
plt.imshow(nd.ones((n,n,3)).asnumpy())
anchors=boxes[20,20,:,:]
for i in range(anchors.shape[0]):
    plt.gca().add_patch(box_to_rect(anchors[i,:]*n,colors[i]))
plt.show()
from mxnet.gluon import nn
def class_predictor(num_anchors,num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors*(num_classes+1),3,padding=1)
cls_pred=class_predictor(5,10)
cls_pred.initialize()
x=nd.zeros((2,3,20,20))
print('Class prediction',cls_pred(x).shape)