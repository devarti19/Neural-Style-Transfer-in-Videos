import os
import cv2
import torch
from style_network import ImageTransformer


def stylize(args):
    #Testing the style transfer on video
    device = torch.device("cuda" if args.cuda else "cpu")

    #load trained model
    model = ImageTransformer()
    model_path = os.path.join(args.save_model, args.model_name)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    #load test image sequence
    test_folder = '.'
    os.chdir(test_folder)
    img_directory = os.path.join(test_folder, args.test_img_path)
    test_imgs = os.listdir(img_directory)
    test_imgs.sort()
    out_video_dir = os.path.join(test_folder, args.output_video)

    #find the image size
    img = cv2.imread(os.path.join(img_directory, test_imgs[0]))
    h, w, c = img.shape
    img_size = (w, h)

    #Stylize images and create a video
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    v_writer = cv2.VideoWriter(out_video_dir, fourcc, args.fps, img_size)

    for i in range(len(test_imgs)):
        t_img = os.path.join(img_directory, test_imgs[i])
        img = cv2.imread(t_img)
        if model is not None:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = torch.from_numpy(img.astype('float32') / 255.0).permute(2, 0, 1).to(device)
            output = model(img.unsqueeze(0))
            output = output.squeeze(0).permute(1, 2, 0)
            concat_img = output.detach().cpu().numpy()
            frame = concat_img * 255
            frame = frame.astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v_writer.write(frame)
    v_writer.release()
    print('Video_Location:', out_video_dir)