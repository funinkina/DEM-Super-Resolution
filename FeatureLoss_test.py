import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import cv2, IOU
import os
#################################
from RiverModel.unet.unet_model import UNet as RiverNet  # unet
from DeviceSetting import device


def write_file(filepath, target_tensor):
    with open(filepath, 'a') as af:
        num_rows, num_cols = target_tensor.shape
        af.write('\n')
        for i in range(num_rows):
            for j in range(num_cols):
                af.write(str(target_tensor[i][j].item()) + ',')
            af.write('\n')


class RiverLoss(nn.Module):
    def __init__(self):
        super(RiverLoss, self).__init__()

    def forward(self, input, targets, base_max_list):
        batchSize, channel, H, W = input.shape

        fake_river_heatmap = unet(input.to(device))
        criterion = nn.BCEWithLogitsLoss().to(device)
        loss = criterion(fake_river_heatmap, targets)
        ############################output miou###########################################
        fake_river = 1.0 * (F.sigmoid(fake_river_heatmap.to(device)).detach().cpu() > 0.5).numpy()
        miou_0 = 0
        miou_0_list = []
        miou_1 = 0
        miou_1_list = []
        Miou = 0
        Miou_list = []
        for j in range(batchSize):
            real_j = np.reshape(targets[j].detach().cpu(), (H, W)).numpy()
            fake_j = np.reshape(fake_river[j].data, (H, W))
            iou = IOU.GetIOU(real_j, fake_j, 2)
            miou_0 = miou_0 + iou[0][0]
            miou_0_list.append(iou[0][0])
            miou_1 = miou_1 + iou[0][1]
            miou_1_list.append(iou[0][1])
            Miou = Miou + np.mean(iou[0])
            Miou_list.append(np.mean(iou[0]))

        miou_0 = miou_0 / batchSize
        miou_1 = miou_1 / batchSize
        Miou = Miou / batchSize
        ####################################################################################
        # return loss, miou_0, miou_1, Miou
        return loss, miou_0_list, miou_1_list, Miou_list


extract_river_Weights = './RiverModel/unet/unet.pth'
unet = RiverNet(n_channels=1, n_classes=1)  # unet
unet.load_state_dict(torch.load(extract_river_Weights, map_location=device, weights_only=True))
unet.to(device)
river_conterion = RiverLoss()
river_conterion.to(device)
