
import numpy as np
import os


# x = np.load('./generated/test/0000056.npy')
# print(x.shape)
# print(x[0].shape)

# from argparse import Namespace
#
# arg = {'hello': 3}
# a = Namespace(**arg)
#
# print(a)
# print(a.hello)


data = np.load('output.npy')
from PIL import Image
data = data * 10
data = data.astype('uint8')
img = Image.fromarray(data)
img.show()