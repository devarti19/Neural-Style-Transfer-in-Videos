import torch

def gram_matrix(image):
    b, c, h, w = image.size()
    lst = []
    for i in range(b):
        x = image[i]
        gram = x.view(c, h * w)
        gram = torch.mm(gram, gram.T)
        lst.append(gram.unsqueeze(0))
    return torch.cat(lst, dim=0) / (c * h * w)

def get_mask(warp_img, sample, mask):

    #relative luminance -  digtal video
    img_gray = 0.2989 * warp_img[:, 2, :, :] + 0.5870 * warp_img[:, 1, :, :] + 0.1140 * warp_img[:, 0, :, :]
    sample_gray = 0.2989 * sample[:, 2, :, :] + 0.5870 * sample[:, 1, :, :] + 0.1140 * sample[:, 0, :, :]

    #relative luminance - real world video
    #img_gray = 0.2126 * warp_img[:, 2, :, :] + 0.7152 * warp_img[:, 1, :, :] + 0.0722 * warp_img[:, 0, :, :]
    #sample_gray = 0.2126 * sample[:, 2, :, :] + 0.7152 * sample[:, 1, :, :] + 0.0722 * sample[:, 0, :, :]

    img_gray = img_gray.unsqueeze(1)
    sample_gray = sample_gray.unsqueeze(1)
    mask_cont = torch.abs(img_gray - sample_gray)
    mask_cont[mask_cont < 0.05] = 0
    mask_cont[mask_cont > 0] = 1
    mask_cont = mask - mask_cont
    mask_cont[mask_cont < 0] = 0
    mask_cont[mask_cont > 0] = 1
    return mask_cont


def warped(img, flow, device):
    b, c, h, w = img.size()
    x = torch.arange(0, w)
    y = torch.arange(0, h)
    y_grid, x_grid = torch.meshgrid(y, x)
    grid = torch.cat((x_grid.unsqueeze(0), y_grid.unsqueeze(0))).repeat(b, 1, 1, 1).float().to(device)
    grid = grid + flow
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (w - 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (h - 1) - 1.0
    grid = grid.permute(0, 2, 3, 1)

    output = torch.nn.functional.grid_sample(img, grid)
    mask_boundary = torch.nn.functional.grid_sample(torch.ones(img.size(), device=device), grid, mode='bilinear')
    mask_boundary[mask_boundary < 0.9999] = 0
    mask_boundary[mask_boundary > 0] = 1
    output = output * mask_boundary
    return output, mask_boundary