import os
import torch
import torchvision
import numpy as np
from PIL import Image
from skimage import transform
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage, Resize

def readFlowfile(name):
    f = open(name, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)


class data_load(Dataset):

    def __init__(self, path):
        #looking at the "clean" subfolder for images.
        #root_path is training folder inside the MPI Sintel dataset folder
        self.width = 512
        self.height = 512
        self.root_path = path
        self.folder_list = os.listdir(self.root_path + "/clean/")
        self.folder_list.sort()
        self.folder_list = [item for item in self.folder_list if item.find('bandage_1') < 0]
        self.imglist = []
        for folder in self.folder_list:
            self.imglist.append(len(os.listdir(self.root_path + "/clean/" + folder + "/")))

    def __len__(self):
        return sum(self.imglist) - len(self.imglist)

    def __getitem__(self, idx):

        for i in range(0, len(self.imglist)):
            folder = self.folder_list[i]
            imgpath = self.root_path + "/clean/" + folder + "/"
            occpath = self.root_path + "/occlusions/" + folder + "/"
            flowpath = self.root_path + "/flow/" + folder + "/"
            if (idx < (self.imglist[i] - 1)):
                n1 = self.converttoString(idx + 1)
                n2 = self.converttoString(idx + 2)
                img1 = Image.open(imgpath + "frame_" + n1 + ".png").resize((self.width, self.height), Image.BILINEAR)
                img2 = Image.open(imgpath + "frame_" + n2 + ".png").resize((self.width, self.height), Image.BILINEAR)
                mask = Image.open(occpath + "frame_" + n1 + ".png").resize((self.width, self.height), Image.BILINEAR)
                flow = readFlowfile(flowpath + "frame_" + n1 + ".flo")
                img1 = ToTensor()(img1).float()
                img2 = ToTensor()(img2).float()
                mask = ToTensor()(mask).float()
                h, w, c = flow.shape
                flow = torch.from_numpy(transform.resize(flow, (self.height, self.width))).permute(2, 0, 1).float()
                # flow[0] contains flow in x direction and flow[1] contains flow in y direction
                flow[0, :, :] = flow[0, :, :] * float(flow.shape[1] / h)
                flow[1, :, :] = flow[1, :, :] * float(flow.shape[2] / w)

                #take no occluded regions to compute
                mask = 1 - mask
                mask[mask < 0.99] = 0
                mask[mask > 0] = 1
                break
            idx = idx - (self.imglist[i] - 1)
        # img2 should be at t in img1 is at t-1
        return (img1, img2, mask, flow)

    def converttoString(self, n):
        string = str(n)
        while (len(string) < 4):
            string = "0" + string
        return string