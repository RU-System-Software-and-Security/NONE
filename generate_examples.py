import torchvision.transforms as transforms
import torch
#print(torch.cuda.device_count())
import torchvision
import torchvision.datasets as datasets
import cv2
import numpy as np
import argparse

def tensor2im(input_image, imtype=np.uint8):
    #print(input_image.shape)
    mean = [0, 0, 0]
    std = [1, 1, 1]
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', default='./example/', type=str, metavar='PATH')
parser.add_argument('--dataset', type=str, default="cifar10")

args = parser.parse_args()


if args.dataset == "cifar10":

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = datasets.CIFAR10
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

counter = [0]*10
for batch_idx, (data, target) in enumerate(testloader):
    print(counter)
    cv2.imwrite(args.save_dir+"class_" + str(target.item()) + "_example_" + str(counter[int(target.item())]) + ".png", tensor2im(data[0,:,:,:]))
    counter[int(target.item())] = counter[int(target.item())] + 1
