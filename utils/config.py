import os

model_savepath = 'checkpoints'
outputs_path = 'outputs'
power = 0.9

cam_w = 100
cam_sigma = 0.4
cam_loss_sigma = 0.75
lse_r = 6.

loss_ano = 0.01
loss_cls = 1
loss_ex_cls = 0.5
loss_bound = 0.001
loss_union = 0.001

cls_thresh = 0.5
cam_thresh = 0.999

thresh_TIOU = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
thresh_TIOR = [0.1, 0.25, 0.5, 0.75, 0.9]

palette = [[0, 0, 0],
           [128, 0, 0],
           [0, 128, 0],
           [128, 128, 0],
           [0, 0, 128],
           [128, 0, 128],
           [0, 128, 128],
           [128, 128, 128],
           [64, 0, 0],
           [192, 0, 0],
           [64, 128, 0],
           [192, 128, 0],
           [64, 0, 128],
           [192, 0, 128],
           [64, 128, 128],
           [192, 128, 128],
           [0, 64, 0],
           [128, 64, 0],
           [0, 192, 0],
           [128, 192, 0],
           [0, 64, 128],
           [224, 224, 192]]
