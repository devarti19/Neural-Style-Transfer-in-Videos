import os
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from torch.optim import Adamax
from dataloader import data_load
import torch.optim.lr_scheduler as ls
from loss_network import Vgg16, Normal
from torch.utils.data import DataLoader
from style_network import ImageTransformer
from utils import gram_matrix, warped, get_mask
style_weight = [1, 1e0, 1e0, 1e0]

def train(args):
    #Training the model
    #code is using GPU if cuda is available else CPU
    device = torch.device("cuda" if args.cuda else "cpu")

    #Building Model
    style_model = ImageTransformer().to(device)
    loss_model = Vgg16().to(device)
    for param in loss_model.parameters():
        param.requires_grad = False

    loss_mse = torch.nn.MSELoss()
    loss_msesum = torch.nn.MSELoss(reduction='none')

    optimizer = Adamax(style_model.parameters(), lr=args.lr)
    schedular = ls.MultiStepLR(optimizer, milestones=[8, 20], gamma=0.2)

    #Loading Data
    train_dataset = data_load(os.path.join(args.dataset, 'training'))
    data_train = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    mean =torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    #Loading Style
    style_img = Image.open(args.style_name)
    style_img = style_img.resize((args.width, args.height), Image.BILINEAR)
    style_img = torchvision.transforms.ToTensor()(style_img)
    style_batch = []
    style_batch.append(style_img)
    normalization = Normal(mean, std)
    count = 0
    while count < args.epochs:
        data = tqdm(data_train)
        count += 1

        for idx, x in enumerate(data):
            optimizer.zero_grad()

            img1, img2, mask, flow = x
            img1, img2 = img2, img1
            img1 = img1.to(device)
            img2 = img2.to(device)
            mask = mask.to(device)
            flow = flow.to(device)
            style_img = style_batch[0].to(device)

            #style network
            output_img1 = style_model(img1)
            output_img2 = style_model(img2)

            #temporal loss
            warp_img1, mask_boundary_img1 = warped(img1, flow, device)
            mask_occ = get_mask(warp_img1, img2, mask)
            warp_output_img1, _ = warped(output_img1, flow, device)
            temporal_loss = loss_msesum(output_img2, warp_output_img1)
            temporal_loss = torch.sum(temporal_loss * mask_occ * mask_boundary_img1) / (
                    img2.size(0) * img2.size(1) * img2.size(2) * img2.size(3))
            temporal_loss *= args.lambda_lt

            #long temporal loss
            if (idx) % 5 == 0:
                frame0 = output_img2

            with torch.no_grad():
                frame0_mask = get_mask(warp_img1, frame0, mask)
                long_temporal_loss = torch.abs(loss_msesum(frame0, warp_output_img1))
                long_temporal_loss = torch.sum(long_temporal_loss * frame0_mask * mask_boundary_img1) / (
                        frame0.size(0) * frame0.size(1) * frame0.size(2) * frame0.size(3))
            long_temporal_loss *= args.lambda_st

            #normalization to vgg16
            img1 = normalization(img1)
            img2 = normalization(img2)
            style_img = normalization(style_img.repeat(output_img1.size(0), 1, 1, 1))
            output_img1 = normalization(output_img1)
            output_img2 = normalization(output_img2)

            #loss network
            content1 = loss_model(img1)
            content2 = loss_model(img2)
            style_out = loss_model(style_img)
            out1 = loss_model(output_img1)
            out2 = loss_model(output_img2)

            #content loss
            loss_content = loss_mse(content1[2], out1[2]) + loss_mse(content2[2], out2[2])
            loss_content *= args.alpha

            #style loss
            loss_style = 0.0
            for i in range(len(style_out)):
                style_gram = gram_matrix(style_out[i])
                out1_gram = gram_matrix(out1[i])
                out2_gram = gram_matrix(out2[i])
                loss_style += style_weight[i] * (loss_mse(style_gram, out1_gram) + loss_mse(style_gram, out2_gram))
            loss_style *= args.beta

            #total variation loss
            loss_tv_img1 = torch.sum(torch.abs(output_img1[:, :, :, :-1] - output_img1[:, :, :, 1:])) \
                      + torch.sum(torch.abs(output_img1[:, :, :-1, :] - output_img1[:, :, 1:, :]))
            loss_tv_img2 = torch.sum(torch.abs(output_img2[:, :, :, :-1] - output_img2[:, :, :, 1:]))\
                           + torch.sum(torch.abs(output_img2[:, :, :-1, :] - output_img2[:, :, 1:, :]))
            loss_tv = (loss_tv_img1 + loss_tv_img2) / output_img1.size(0)
            loss_tv *= args.gamma

            data.set_description('Epoch:%d temp_loss:%.7f long_temp_loss:%.7f content_loss:%.2f '
                                 'style_loss:%.7f tv_loss:%.1f'% 
                                  (count, temporal_loss.item(),
                                   long_temporal_loss.item(),
                                   loss_content.item(),
                                   loss_style.item(),
                                   loss_tv.item()))

            loss = loss_content + loss_style + loss_tv + temporal_loss + long_temporal_loss

            #backpropogation
            loss.backward()
            optimizer.step()

            if (args.schedular):
                schedular.step()
    if (not os.path.exists(args.save_model)):
        os.mkdir(args.save_model)
    model_path = os.path.join(args.save_model, 'train.pt')
    torch.save(style_model.state_dict(), model_path)









