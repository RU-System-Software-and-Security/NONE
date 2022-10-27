import numpy as np
import torch
import csv

class HandleCsv:
    # 定义存放csv内容的list
    csv_list = []

    def __init__(self, filename):
        self.filename = filename
        with open(self.filename)as fp:
            self.csv_list = list(csv.reader(fp))
            #print(self.csv_list)

    # 在第N行第M列空白单元格处修改内容
    def modify(self, n, m, value):
        self.csv_list[n - 1][m - 1] = value

    # 插入第N行
    def insert_row(self, n):
        self.csv_list.insert(n - 1, [])

    # 在第N行第M列单元格插入
    def insert_col(self, n, m, value):
        # 如果该单元格左边的单元格为空，那么先对左边的单元格写入空格
        if len(self.csv_list[n - 1]) < m:
            if len(self.csv_list[n - 1]) == m - 1:
                self.csv_list[n - 1].append(value)
            else:
                for i in range(len(self.csv_list[n - 1]), m - 1):
                    self.csv_list[n - 1].append('')
                self.csv_list[n - 1].append(value)
        else:
            self.modify(n, m, value)

    # 删除第N行
    def del_row(self, n):
        del self.csv_list[n - 1]

    # 获取第n行第m列单元格内容
    def get_value(self, n, m):
        return self.csv_list[n - 1][m - 1]

    def list2csv(self, file_path):
        try:
            fp = open(file_path, 'w')
            for items in self.csv_list:
                for i in range(len(items)):
                    # 若元素中含有逗号，那么需要加双引号
                    if items[i].find(',') != -1:
                        fp.write('\"')
                        fp.write(items[i])
                        fp.write('\"')
                    else:
                        fp.write(items[i])
                    # 最后一个元素不用加逗号
                    if i < len(items) - 1:
                        fp.write(',')
                fp.write('\n')
        except Exception as e:
            print(e)

def tensor2im(input_image, imtype=np.uint8):
    #print(input_image.shape)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
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
    
def tensor2im_gtsrb(input_image, imtype=np.uint8):
    #print(input_image.shape)
    mean = [0.3403, 0.3121, 0.3214]
    std = [0.2724, 0.2608, 0.2669]
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
    
def tensor2im_imagenette(input_image, imtype=np.uint8):
    #print(input_image.shape)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
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

def tensor2im_mnist(input_image, imtype=np.uint8):
    #print(input_image.shape)
    mean = [0.1307]
    std = [0.3081]
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()
        '''if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))'''
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)
    

def _to_one_hot(labels,num_classes):
    one_hot = torch.zeros((labels.size(0), num_classes), requires_grad=True)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1

    one_hot = torch.autograd.Variable(one_hot)

    return one_hot

class generated_dataset():
    def __init__(self, data, label):

        self.train_data = data
        self.train_labels = label

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        return img, target