import os
import torch
import pickle
import numpy
import torchvision.transforms as transforms
from PIL import Image
import albumentations
def addnoise(img,p=1,mean=0,var_limit=(0,1)):
    #img = np.asarray(img)
    aug = albumentations.GaussNoise(p=p,mean=mean,var_limit=var_limit)
    augmented = aug(image=(img).astype(numpy.uint8))
    auged = augmented['image']
    #auged = Image.fromarray(auged)
    return auged
    
class dataset():
    def __init__(self, root=None, train=True, transform=None, all_noise=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.all_noise = all_noise
        
        
        if self.train:
            train_data_path = os.path.join(root, 'train_images.npy')
            train_labels_path = os.path.join(root, 'train_labels.npy')
            self.train_data = numpy.load(open(train_data_path, 'rb'))
            #self.train_data = torch.from_numpy(self.train_data.astype('float32'))
            #self.train_data = numpy.transpose(self.train_data, (0,3,1,2))
            #self.train_data = Image.fromarray(self.train_data)
            self.train_labels = numpy.load(open(train_labels_path, 'rb')).astype('int')
        else:
            test_data_path = os.path.join(root, 'test_images.npy')
            test_labels_path = os.path.join(root, 'test_labels.npy')
            self.test_data = numpy.load(open(test_data_path, 'rb'))
            #self.test_data = torch.from_numpy(self.test_data.astype('float32'))
            #self.test_data = numpy.transpose(self.test_data, (0,3,1,2))
            #self.test_data = Image.fromarray(self.test_data)
            self.test_labels = numpy.load(open(test_labels_path, 'rb')).astype('int')
            #print(self.test_data.shape)
            #print(self.test_labels)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            if self.transform:
                #print(self.train_data.shape)
                img, target = self.transform(Image.fromarray(numpy.uint8(self.train_data[index]))), self.train_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            if self.transform:
                #img, target = self.transform(Image.fromarray(numpy.uint8(self.test_data[index]))), self.test_labels[index]
                img = numpy.uint8(self.test_data[index])
                if self.all_noise:
                    img = addnoise(img,p=1,mean=80,var_limit=(0,80))
                img = self.transform(Image.fromarray(img))
                target = self.test_labels[index]
                
            else:
                img, target = self.test_data[index], self.test_labels[index]

        

        return img, target
