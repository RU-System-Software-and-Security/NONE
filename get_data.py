from data import utils,npy_data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
print(torch.cuda.device_count())
def get_dataloaders(dataset_name,poison_type,trigger_size,poison_rate):

    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if poison_type == "none":
            dataloader = datasets.CIFAR10
            trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            
            trainset_transform_test = dataloader(root='./data', train=True, download=True, transform=transform_test)
            trainloader_no_shuffle = torch.utils.data.DataLoader(trainset_transform_test, batch_size=128, shuffle=False, num_workers=4)
            trainloader_no_shuffle_bs1 = torch.utils.data.DataLoader(trainset_transform_test, batch_size=1, shuffle=False)
            
            testset = npy_data.dataset(root='./dirty_label_p05_s3', train=False, transform=transform_test)
            testloader_poisoned = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

        elif poison_type == "label_specific":
            if poison_rate == 0.05:        
                dataloader = datasets.CIFAR10
                trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
                trainloader_clean = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            
                trainset = npy_data.dataset(root='./label_specific_p5_s3', train=True, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

                trainset_transform_test = npy_data.dataset(root='./label_specific_p5_s3', train=True, transform=transform_test)
                trainloader_no_shuffle = torch.utils.data.DataLoader(trainset_transform_test, batch_size=128, shuffle=False, num_workers=4)
                trainloader_no_shuffle_bs1 = torch.utils.data.DataLoader(trainset_transform_test, batch_size=1, shuffle=False)

                testset = npy_data.dataset(root='./label_specific_p5_s3', train=False, transform=transform_test)
                testloader_poisoned = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

        elif poison_type == "single_target":

            if (trigger_size == 3) and (poison_rate==0.05):
                trainset = npy_data.dataset(root='./dirty_label_p5_s3', train=True, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
                
                trainset_transform_test = npy_data.dataset(root='./dirty_label_p5_s3', train=True, transform=transform_test)
                trainloader_no_shuffle = torch.utils.data.DataLoader(trainset_transform_test, batch_size=128, shuffle=False, num_workers=4)
                trainloader_no_shuffle_bs1 = torch.utils.data.DataLoader(trainset_transform_test, batch_size=1, shuffle=False)
                
                testset = npy_data.dataset(root='./dirty_label_p5_s3', train=False, transform=transform_test)
                testloader_poisoned = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

        elif poison_type == "clean_label":
            dataloader = datasets.CIFAR10
            trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
            trainloader_clean = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        
            trainset = npy_data.dataset(root='./clean_label_two1200_p5_s3', train=True, transform=transform_train)#acctually is 0.005!
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            
            trainset_transform_test = npy_data.dataset(root='./clean_label_two1200_p5_s3', train=True, transform=transform_test)
            trainloader_no_shuffle = torch.utils.data.DataLoader(trainset_transform_test, batch_size=128, shuffle=False, num_workers=4)
            trainloader_no_shuffle_bs1 = torch.utils.data.DataLoader(trainset_transform_test, batch_size=1, shuffle=False)
            
            testset = npy_data.dataset(root='./clean_label_two1200_p5_s3', train=False, transform=transform_test)
            testloader_poisoned = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
            
        else:
            print("poison_type is not supported")
            

        dataloader = datasets.CIFAR10
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        img_size=32
        num_classes = 10
    else:
        print("dataset is not supported")
        
    return num_classes,trainset,trainloader,testloader,testloader_poisoned,trainloader_no_shuffle