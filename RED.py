# Import stuff
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import os

rand = np.random.uniform



# shape = image.shape
# shape_size = shape[:2]

shape_size = [480, 480]
nodenum = 7
# offset_max = shape_size[0] / nodenum / 100
offset_max = shape_size[0] * 1/18
meshnode_mat_y = np.zeros((nodenum + 2, nodenum + 2))
meshnode_mat_x = np.zeros((nodenum + 2, nodenum + 2))
meshnode_mat_y[1:-1, 1:-1] = rand(-offset_max, offset_max, (nodenum ,nodenum ))
meshnode_mat_x[1:-1, 1:-1] = rand(-offset_max, offset_max, (nodenum, nodenum))
# print(meshnode_mat_y)
ind = np.meshgrid(np.arange(shape_size[0]), np.arange(shape_size[1]))
# ind_inter = [ind[0] / shape_size[0]* (nodenum + 1), ind[1] / shape_size[1]* (nodenum + 1)]
ind_inter = np.meshgrid(np.linspace(0, nodenum + 1, shape_size[0]), np.linspace(0, nodenum + 1, shape_size[1]))
# print(ind_inter[0])
# print(ind[0])
mat_y = map_coordinates(meshnode_mat_y,ind_inter,order=3, mode='reflect')
mat_x = map_coordinates(meshnode_mat_x,ind_inter,order=3, mode='reflect')
# print(mat_y)
# print(mat_y)

plt.figure()
plt.imshow(mat_x)
plt.figure()
plt.imshow(meshnode_mat_x)
# plt.show()


im = cv2.imread('/home/zhaojin/Pictures/building.jpeg', cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, (480, 480))
print(im.shape)
# im = np.zeros([480, 480])
# im[0:-1:40, 0:-1:40] = 1
# im = gaussian_filter(im, sigma=1)
# th = 0.01
# im[im >= th] = 255
# im[im < th] = 0
img_ind = [mat_x.reshape(-1) + ind[0].reshape(-1), mat_y.reshape(-1) + ind[1].reshape(-1)]
im_new = map_coordinates(im, img_ind, order=1, mode='reflect').reshape(shape_size[0], shape_size[1]).transpose((1,0))
plt.figure()

plt.imshow(im_new)
plt.figure()
plt.imshow(im)
plt.show()