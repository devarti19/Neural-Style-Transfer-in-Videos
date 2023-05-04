import os
import math
import torch
from tqdm import tqdm
from utils import warped
from dataloader import data_load
from torch.utils.data import DataLoader

def estab(args):
    #Calculating the temporal error in stylized video
    device = torch.device("cuda" if args.cuda else "cpu")
    total_list = []
    output_img = data_load(os.path.join(args.path, 'alley-2'))
    data_output = DataLoader(dataset=output_img, batch_size=1, shuffle=True)
    data = tqdm(data_output)
    loss_msesum = torch.nn.MSELoss(reduction='none')

    for idx, x in enumerate(data):
        img1, img2, mask, flow = x
        img1, img2 = img2, img1
        img1 = img1.to(device)
        img2 = img2.to(device)
        mask = mask.to(device)
        flow = flow.to(device)

        warp_img1, _ = warped(img1, flow)
        temp_output_loss = loss_msesum(img2, warp_img1)
        temp_output_loss = torch.sum((temp_output_loss * mask) / (
                img2.size(0) * img2.size(1) * img2.size(2) * img2.size(3)))
        total_list.append(temp_output_loss.tolist())
    estab = sum(total_list)
    estab = math.sqrt(estab/49)
    print(estab)