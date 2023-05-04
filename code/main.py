import argparse
from train import train
from test import stylize
from evaluate import estab

def get_args_parser():
    main_arg_parser = argparse.ArgumentParser(description="parser for video_style_transfer")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--cuda", type=int, required=True,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--width", type=int, default=512,
                              help="width of input image")
    train_parser.add_argument("--height", type=int, default=512,
                              help="height of input image")
    train_parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                              help='input batch size for training (default: 2)')
    train_parser.add_argument('--epochs', type=int, default=100,
                              help='epoch')
    train_parser.add_argument('--lr', type=float, default=0.0001,
                              help='learning rate')
    train_parser.add_argument("--dataset", type=str, required=True,
                              help="path to image dataset")
    train_parser.add_argument("--style_name", type=str, required=True,
                              help="path to a style image to train with")
    train_parser.add_argument('--mean', type=float, default=[0.485, 0.456, 0.406],
                              help='Mean for Normalization for VGG network')
    train_parser.add_argument('--std', type=float, default=[0.229, 0.224, 0.225],
                              help='Standard Deviation for Normalization for VGG network')
    train_parser.add_argument('--alpha', type=float, default=1e0,
                              help='Content Loss weight')
    train_parser.add_argument('--beta', type=float, default=1e5,
                              help='Style Loss weight')
    train_parser.add_argument('--gamma', type=float, default=1e-6,
                              help='Total Variation weight')
    train_parser.add_argument('--lambda_st', type=float, default=1e2,
                              help='Temporal loss weight')
    train_parser.add_argument('--lambda_lt', type=float, default=1e1,
                              help='Long Temporal loss weight')
    train_parser.add_argument('--schedular', type=bool, default=True,
                              help='schedular')
    train_parser.add_argument('--save_model', type=str, default='trained_models',
                              help='trained model is saved here')


    style_parser = subparsers.add_parser("transfer", help="style transfer with a trained model")
    style_parser.add_argument("--cuda", type=int, required=True,
                              help="set it to 1 for running on GPU, 0 for CPU")
    style_parser.add_argument('--save_model', type=str, default='trained_models',
                              help='trained model is saved here')
    style_parser.add_argument('--model-name', type=str, default='',
                              help='model name')
    style_parser.add_argument('--test_img_path', type=str, default='',
                              help='Path for test images')
    style_parser.add_argument('--output_video', type=str, default='output\output.avi', metavar='N',
                              help='Output video name')
    style_parser.add_argument('--fps', type=int, default=15, metavar='N',
                              help='input batch size for training')

    evaluate_parser = subparsers.add_parser("evaluate", help="Calculate temporal error")
    evaluate_parser.add_argument("--path", type=str, required=True,
                              help="path to output images")
    evaluate_parser.add_argument("--cuda", type=int, required=True,
                              help="set it to 1 for running on GPU, 0 for CPU")

    return main_arg_parser.parse_args()

def main():
    args = get_args_parser()
    # command
    if (args.subcommand == "train"):
        train(args)
    elif (args.subcommand == "transfer"):
        stylize(args)
    elif (args.subcommand == "evaluate"):
        estab(args)
    else:
        print("invalid_command")

if __name__ == '__main__':
    main()