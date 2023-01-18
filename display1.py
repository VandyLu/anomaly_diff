import cv2
import numpy as np


filename = 'work_dirs/samples_32x64x64x3.npz'
filename = 'work_dirs/samples_32x256x256x3.npz'
# filename = 'work_dirs/samples_32x128x128x3.npz'
filename = 'work_dirs_128/samples_32x128x128x3.npz'
filename = 'work_dirs/pill_5k_2gpu/samples_100x128x128x3.npz'
filename = 'work_dirs/capsule_test/samples_100x128x128x3.npz'

mid_filename = 'work_dirs/mid_samples_32x128x128x3.npz'
mid_filename = 'work_dirs/cable_test/samples_100x128x128x3.npz'

vis_mid = False
if vis_mid:
    outputs = np.load(mid_filename)['arr_0']
    for i in range(outputs.shape[0]):
        img_i = outputs[i][..., ::-1] # BGR -> RGB
        for j in range(img_i.shape[0]):
            name = './vis_{:05d}_{:03d}.png'.format(i, j)
            cv2.imwrite(name, img_i[j])
        print(name, img_i.shape, img_i.max())
else:
    outputs = np.load(filename)['arr_0']
    for i in range(outputs.shape[0]):
        img_i = outputs[i][..., ::-1] # BGR -> RGB
        name = './vis_{:05d}.png'.format(i)
        cv2.imwrite(name, img_i)
        print(name, img_i.shape)