import numpy as np
import os
import argparse

import sys
import json
import skimage.io
import random
import torch
import torch.nn.functional as F
import pickle
import time
import torch.nn as nn

from models import nin,resnet,vgg,vgg_trojai

np.set_printoptions(precision=2, linewidth=200, threshold=10000)

def seed_torch(seed=333):#333

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch()
print(torch.rand(1))
print(torch.rand(1))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

            )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), 10)
        return x



# with open('./config.json') as config_file:
#     config = json.load(config_file)
config = {}
config['gpu_id'] = '0'
config['print_level'] = 2
config['random_seed'] = 333
config['num_classes'] = 10
config['channel_last'] = 0
config['w'] = 32
config['h'] = 32
config['reasr_bound'] = 0.2
config['batch_size'] = 20
config['has_softmax'] = 0
config['samp_k'] = 1
#config['samp_k'] = 5
config['same_range'] = 0
config['n_samples'] = 3
config['samp_batch_size'] = 8
config['top_n_neurons'] = 10
#config['top_n_neurons'] = 100
config['re_batch_size'] = 80
#config['max_troj_size'] = 64
config['max_troj_size'] = 64
config['filter_multi_start'] = 1
config['re_mask_lr'] = 0.1
#config['re_mask_lr'] = 5e-1
config['re_mask_weight'] = 50
config['mask_multi_start'] = 1
config['re_epochs'] = 50
config['n_re_samples'] = 240

#score_type = "max"
score_type = "mean_exclude_target"

#config['re_epochs'] = 50
#config['n_re_samples'] = 10000

#config['n_re_samples'] = 1000

#config['re_epochs'] = 100
#config['n_re_samples'] = 10000


channel_last = bool(config['channel_last'])
random_seed = int(config['random_seed'])
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]

#np.random.seed(random_seed)
#seed_torch(random_seed)
w = config["w"]
h = config["h"]
num_classes   = config["num_classes"]
use_mask = True
count_mask = True
tdname = 'temp'
window_size = 12
mask_epsilon = 0.01
delta_shape = [window_size,window_size,3,3]
Troj_size = config['max_troj_size']
reasr_bound = float(config['reasr_bound'])
top_n_neurons = int(config['top_n_neurons'])
mask_multi_start = int(config['mask_multi_start'])
filter_multi_start = int(config['filter_multi_start'])
re_mask_weight = float(config['re_mask_weight'])
re_mask_lr = float(config['re_mask_lr'])
batch_size = config['batch_size']
has_softmax = bool(config['has_softmax'])
print('channel_last', channel_last, 'gpu_id', config["gpu_id"], 'has softmax', has_softmax)
# try_epochs = 10

# cifar= CIFAR10()
# print('gpu id', config["gpu_id"])
# l_bounds = cifar.l_bounds
# h_bounds = cifar.h_bounds
# print('mean', cifar.mean, 'std', cifar.std, 'l bounds', l_bounds[0,0,0], 'h_bounds', h_bounds[0,0,0])

# l_bounds_channel_first = np.transpose(l_bounds, [0,3,1,2])
# h_bounds_channel_first = np.transpose(h_bounds, [0,3,1,2])
Print_Level = int(config['print_level'])
re_epochs = int(config['re_epochs'])
n_re_samples = int(config['n_re_samples'])

arch = None

def preprocess(img):
    img = np.transpose(img, [0,3,1,2])
    return img.astype(np.float32) / 255.0

'''def deprocess(img):
    img = np.transpose(img, [0,2,3,1])
    return (img*255).astype(np.uint8)'''

def deprocess(img):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    img_copy = img.copy()
    for j in range(img_copy.shape[0]):
        for i in range(len(mean)):
            #img[i] = (img[i] - mean[i])/std[i]
            img_copy[j][i] = img_copy[j][i]*std[i] + mean[i]
    img_copy = np.transpose(img_copy, [0,2,3,1])
    return (img_copy*255).astype(np.uint8)


def check_values(images, labels, model, children, target_layers):
    maxes = {}
    print("check_values")
    #print(images)
    for layer_i in range(0, len(children) - 1):
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        temp_model1 = torch.nn.Sequential(*children[:layer_i+1])

        max_val = -np.inf
        print(len(images))
        print(batch_size)
        for i in range(len(images)//batch_size):
            batch_data = torch.FloatTensor(images[batch_size*i:batch_size*(i+1)])
            batch_data = batch_data.cuda()
            inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
            if channel_last:
                n_neurons = inner_outputs.shape[-1]
            else:
                n_neurons = inner_outputs.shape[1]
            max_val = np.maximum(max_val, np.amax(inner_outputs))
            print(np.amax(inner_outputs))
        
        key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)
        maxes[key] = [max_val]
        print('max val', key, max_val)
        del temp_model1, batch_data, inner_outputs
    return maxes

def sample_neuron(images, labels, model, children, target_layers, model_type, mvs):
    all_ps = {}
    samp_k = config['samp_k']
    same_range = config['same_range']
    n_samples = config['n_samples']
    sample_batch_size = config['samp_batch_size']
    if model_type == 'DenseNet':
        sample_batch_size = max(sample_batch_size // 3, 1)
    n_images = images.shape[0]
    if Print_Level > 0:
        print('sampling n imgs', n_images)

    end_layer = len(children)-1
    if has_softmax:
        end_layer = len(children)-2

    # maxv_layer_i = 0
    # fmaxv = -np.inf
    # for layer_i in range(2, end_layer):
    #     if not children[layer_i].__class__.__name__ in target_layers:
    #         continue
    #     mv_key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)
    #     maxv = max(mvs[mv_key])
    #     if maxv > fmaxv:
    #         fmaxv = maxv
    #         maxv_layer_i = layer_i
    # sample_layers = [maxv_layer_i]

    sample_layers = []
    for layer_i in range(2, end_layer):
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        sample_layers.append(layer_i)
    #sample_layers = sample_layers[-3:-1]
    if arch == "nin":
        sample_layers = sample_layers[-2:-1]
    elif arch == "lenet":
        sample_layers = sample_layers[-2:-1]
    elif arch == "vgg16":
        sample_layers = sample_layers[-1:]
    elif arch == "vgg11":
        sample_layers = sample_layers[-1:]
    else:
        sample_layers = sample_layers[-1:]
    
    #sample_layers = sample_layers[-2:-1]
    
    # sample_layers = sample_layers

    for layer_i in sample_layers:
        #print("layer_i",layer_i)
        #if Print_Level > 0:
            #print('layer', layer_i, children[layer_i])
        temp_model1 = torch.nn.Sequential(*children[:layer_i+1])
        if has_softmax:
            temp_model2 = torch.nn.Sequential(*children[layer_i+1:-1])
        else:
            temp_model2 = torch.nn.Sequential(*children[layer_i+1:])

        if same_range:
            vs = np.asarray([i*samp_k for i in range(n_samples)])
        else:
            mv_key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)

            # tr = samp_k * max(mvs[mv_key])/(n_samples - 1)
            # vs = np.asarray([i*tr for i in range(n_samples)])

            maxv = max(mvs[mv_key])
            e_scale = np.array([0] + [np.power(2., i-1) for i in range(n_samples)])
            vs = max(mvs[mv_key]) * e_scale

            #print('mv_key', vs)
        
        for input_i in range(n_images//batch_size):
            #print("input_i",input_i)
            batch_data = torch.FloatTensor(images[batch_size*input_i:batch_size*(input_i+1)])
            batch_data = batch_data.cuda()
            inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
            if channel_last:
                n_neurons = inner_outputs.shape[-1]
            else:
                n_neurons = inner_outputs.shape[1]

            nbatches = n_neurons//sample_batch_size
            #print(nbatches)
            for nt in range(nbatches):
                #print("nt",nt)
                l_h_t = []
                for neuron in range(sample_batch_size):
                    # h_t = torch.repeat_interleave(inner_outputs, repeats=n_samples, dim=0)
                    if len(inner_outputs.shape) == 4:
                        h_t = np.tile(inner_outputs, (n_samples, 1, 1, 1))
                    else:
                        h_t = np.tile(inner_outputs, (n_samples, 1))

                    for i,v in enumerate(vs):
                        if len(inner_outputs.shape) == 4:
                            if channel_last:
                                h_t[i*batch_size:(i+1)*batch_size,:,:,neuron+nt*sample_batch_size] = v
                            else:
                                h_t[i*batch_size:(i+1)*batch_size,neuron+nt*sample_batch_size,:,:] = v
                        else:
                            h_t[i*batch_size:(i+1)*batch_size, neuron+nt*sample_batch_size] = v
                    l_h_t.append(h_t)
                # f_h_t = torch.cat(l_h_t, axis=0)
                f_h_t = np.concatenate(l_h_t, axis=0)

                f_h_t_t = torch.FloatTensor(f_h_t).cuda()
                print(temp_model2)
                print(f_h_t_t.shape)
                print(temp_model2[0](f_h_t_t).shape)
                print(temp_model2[:2](f_h_t_t).shape)
                print(temp_model2[:3](f_h_t_t).shape)
                fps = temp_model2(f_h_t_t).cpu().detach().numpy()
                # if Print_Level > 1:
                #     print(nt, n_neurons, 'inner_outputs', inner_outputs.shape, 'f_h_t', f_h_t.shape, 'fps', fps.shape)
                for neuron in range(sample_batch_size):
                    tps = fps[neuron*n_samples*batch_size:(neuron+1)*n_samples*batch_size]
                    for img_i in range(batch_size):
                        img_name = (labels[img_i + batch_size*input_i], img_i + batch_size*input_i)
                        ps_key= (img_name, '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i), neuron+nt*sample_batch_size)
                        ps = [tps[ img_i +batch_size*_] for _ in range(n_samples)]
                        ps = np.asarray(ps)
                        ps = ps.T
                        # print('img i', img_i, input_i, batch_size, 'neuron', neuron, ps_key, ps.shape)
                        all_ps[ps_key] = np.copy(ps)

                if input_i == 0 and nt == 0:
                    os.system('nvidia-smi')
                del f_h_t_t
            del batch_data, inner_outputs
            torch.cuda.empty_cache()

        del temp_model1, temp_model2
    return all_ps, sample_layers


def find_min_max(model_name, all_ps, sample_layers, cut_val=20, top_k = 10):
    max_ps = {}
    max_vals = []
    n_classes = 0
    n_samples = 0
    for k in sorted(all_ps.keys()):
        #print(k)
        all_ps[k] = all_ps[k][:, :cut_val]
        n_classes = all_ps[k].shape[0]
        n_samples = all_ps[k].shape[1]
        # maximum increase diff

        vs = []
        for l in range(num_classes):
            vs.append( np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) )
            # vs.append( np.amax(all_ps[k][l]) - np.amin(all_ps[k][l]) )
            # vs.append( np.amax(all_ps[k][l][all_ps[k].shape[1]//5:]) - np.amin(all_ps[k][l][:all_ps[k].shape[1]//5]) )
        ml = np.argsort(np.asarray(vs))[-1]
        sml = np.argsort(np.asarray(vs))[-2]
        val = vs[ml] - vs[sml]
        # print(k, ml, sml)

        max_vals.append(val)
        max_ps[k] = (ml, val)
    
    neuron_ks = []
    imgs = []
    for k in sorted(max_ps.keys()):
        #print(k)
        nk = (k[1], k[2])
        neuron_ks.append(nk)
        imgs.append(k[0])
    #print(neuron_ks)
    neuron_ks = list(set(neuron_ks))
    neuron_ks = sorted(neuron_ks,reverse=True)
    imgs = list(set(imgs))
    #print('imgs', imgs)
    
    min_ps = {}
    min_vals = []
    n_imgs = len(imgs)
    
    print(neuron_ks)
    
    for k in neuron_ks:
        vs = []
        ls = []
        vdict = {}
        for img in sorted(imgs):
            nk = (img, k[0], k[1])
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            vs.append(v)
            ls.append(l)
            if not ( l in vdict.keys() ):
                vdict[l] = [v]
            else:
                vdict[l].append(v)
        ml = max(set(ls), key=ls.count)


        fvs = []
        # does not count when l not equal ml
        for img in sorted(imgs):
            img_l = int(img[0])
            if img_l == ml:
                continue
            nk = (img, k[0], k[1])
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            if l != ml:
                continue
            fvs.append(v)

        # if ml == 2:
        #     print(ml, k, len(fvs), np.amin(fvs), fvs)
        
        if len(fvs) > 0:
            min_ps[k] = (ml, ls.count(ml), np.amin(fvs), fvs)
            min_vals.append(np.amin(fvs))
            # min_ps[k] = (ml, ls.count(ml), np.mean(fvs), fvs)
            # min_vals.append(np.mean(fvs))

        else:
            min_ps[k] = (ml, 0, 0, fvs)
            min_vals.append(0)
    
    
    keys = min_ps.keys()
    keys = []
    for k in min_ps.keys():
        print(k)
        print(int(n_imgs * 0.9))
        print(min_ps[k][1])
        #if min_ps[k][1] >= int(n_imgs * 0.9):
        if arch == "vgg11bn":
            if min_ps[k][1] >= int(n_imgs * 0.7):
                keys.append(k)
        else:
            if min_ps[k][1] >= int(n_imgs * 0.9):
                keys.append(k)
    sorted_key = sorted(keys, key=lambda x: min_ps[x][2] )
    
    print(sorted_key)
    
    if Print_Level > 0:
        print('n samples', n_samples, 'n class', n_classes, 'n_imgs', n_imgs)
        # print('sorted_key', sorted_key)


    neuron_dict = {}
    neuron_dict[model_name] = []
    maxval = min_ps[sorted_key[-1]][2]
    layers = {}
    labels = {}
    allns = 0
    max_sampling_val = -np.inf

    # for i in range(len(sorted_key)):
    #     k = sorted_key[-i-1]
    #     layer = k[0]
    #     neuron = k[1]
    #     label = min_ps[k][0]
    #     if layer not in layers.keys():
    #         layers[layer] = 1
    #     else:
    #         layers[layer] += 1
    #     if layers[layer] <= 3:
    #         if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
    #             continue
    #         if min_ps[k][2] > max_sampling_val:
    #             max_sampling_val = min_ps[k][2]
    #         if Print_Level > 0:
    #             print('min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
    #             if Print_Level > 1:
    #                 print(min_ps[k][3])
    #         allns += 1
    #         neuron_dict[model_name].append( (layer, neuron, min_ps[k][0]) )
    #     if allns >= top_k:
    #     # if allns > top_k//2:
    #         break

    # early layers
    labels = {}
    for i in range(len(sorted_key)):
        k = sorted_key[-i-1]
        layer = k[0]
        neuron = k[1]
        label = min_ps[k][0]
        if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
            continue
        if label not in labels.keys():
            labels[label] = 0
        if int(layer.split('_')[-1]) < sample_layers[-1] and labels[label] < 1:
        # if True:
            labels[label] += 1

            if min_ps[k][2] > max_sampling_val:
                max_sampling_val = min_ps[k][2]
            if Print_Level > 0:
                print('min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                if Print_Level > 1:
                    print(min_ps[k][3])
            allns += 1
            neuron_dict[model_name].append( (layer, neuron, min_ps[k][0]) )
        if allns >= top_k:
            break

    # last layers
    labels = {}
    for i in range(len(sorted_key)):
        k = sorted_key[-i-1]
        layer = k[0]
        neuron = k[1]
        label = min_ps[k][0]
        if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
            continue
        if label not in labels.keys():
            labels[label] = 0
        if int(layer.split('_')[-1]) == sample_layers[-1] and labels[label] < 1:
        # if True:
            labels[label] += 1

            if min_ps[k][2] > max_sampling_val:
                max_sampling_val = min_ps[k][2]
            if Print_Level > 0:
                print('min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                if Print_Level > 1:
                    print(min_ps[k][3])
            allns += 1
            neuron_dict[model_name].append( (layer, neuron, min_ps[k][0]) )
        if allns >= top_k:
            break

    return neuron_dict, max_sampling_val

def read_all_ps(model_name, all_ps, sample_layers, top_k=10, cut_val=20):
    return find_min_max(model_name, all_ps, sample_layers,  cut_val, top_k=top_k)

def filter_img():
    mask = np.zeros((h, w), dtype=np.float32)
    Troj_w = int(np.sqrt(Troj_size) * 0.8) 
    for i in range(h):
        for j in range(w):
            if j >= h/2 and j < h/2 + Troj_w \
                and i >= w/2 and  i < w/2 + Troj_w:
                mask[j,i] = 1
    return mask


def nc_filter_img():
    if use_mask:
        mask = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                # if not( j >= w*1/4.0 and j < w*3/4.0  and i >= h*1/4.0 and i < h*3/4.0):
                if True:
                    mask[i,j] = 1
        mask = np.zeros((h, w), dtype=np.float32) + 1
    else:
        mask = np.zeros((h, w), dtype=np.float32) + 1
    return mask


def loss_fn(inner_outputs_b, inner_outputs_a, logits, con_mask, neuron, tlabel, acc, e, re_epochs):
    neuron_mask = torch.zeros([1, inner_outputs_a.shape[1],1,1]).cuda()
    neuron_mask[:,neuron,:,:] = 1
    vloss1     = torch.sum(inner_outputs_b * neuron_mask)/torch.sum(neuron_mask)
    vloss2     = torch.sum(inner_outputs_b * (1-neuron_mask))/torch.sum(1-neuron_mask)
    relu_loss1 = torch.sum(inner_outputs_a * neuron_mask)/torch.sum(neuron_mask)
    relu_loss2 = torch.sum(inner_outputs_a * (1-neuron_mask))/torch.sum(1-neuron_mask)

    vloss3     = torch.sum(inner_outputs_b * torch.lt(inner_outputs_b, 0) )/torch.sum(1-neuron_mask)

    loss = - vloss1 - relu_loss1  + 0.0001 * vloss2 + 0.0001 * relu_loss2
    #print(vloss1,relu_loss1,vloss2,relu_loss2)
    # if vloss1 < 0 and e < try_epochs:
    #     print('use vloss3', e)
    #     loss = - vloss3# - vloss1
    # else:
    #     print('not use vloss3', e)
    mask_loss = torch.sum(con_mask)
    mask_nz = torch.sum(torch.gt(con_mask, mask_epsilon))
    mask_cond1 = torch.gt(mask_nz, Troj_size)
    mask_cond2 = torch.gt(mask_nz, Troj_size * 1.2)
    # mask_cond1 = torch.gt(mask_loss, Troj_size)
    # mask_cond2 = torch.gt(mask_loss, Troj_size * 1.2)
    mask_add_loss = torch.where(mask_cond1, torch.where(mask_cond2, 2 * re_mask_weight * mask_loss, 1 * re_mask_weight * mask_loss), 0.01 * mask_loss)
    loss += mask_add_loss
    logits_loss = torch.sum(logits[:,tlabel])
    loss += - 1 * logits_loss
    
    print("vloss1",vloss1,"\nrelu_loss1",relu_loss1,"\nvloss2",vloss2,"\nrelu_loss2",relu_loss2,"\nmask_loss",mask_loss,"\nmask_add_loss",mask_add_loss,"\nlogits_loss",logits_loss,"\n")
    
    if e > re_epochs//2:
    # if True:
        loss += - 2 * logits_loss
    return loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, mask_loss, mask_nz, mask_add_loss, logits_loss
    
def reverse_engineer(model_type, model, children, oimages, olabels, weights_file, Troj_Layer, Troj_Neuron, Troj_Label, Troj_size, re_epochs):
    
    before_block = []
    def get_before_block():
        def hook(model, input, output):
            for ip in input:
                before_block.append( ip.clone() )
        return hook
    
    after_bn3 = []
    def get_after_bn3():
        def hook(model, input, output):
            for ip in output:
                after_bn3.append( ip.clone() )
        return hook
    
    after_iden = []
    def get_after_iden():
        def hook(model, input, output):
            for ip in output:
                after_iden.append( ip.clone() )
        return hook

    after_bns = []
    def get_after_bns():
        def hook(model, input, output):
            for ip in output:
                after_bns.append( ip.clone() )
        return hook


    re_batch_size = config['re_batch_size']
    if model_type == 'DenseNet':
        re_batch_size = max(re_batch_size // 3, 1)
    if model_type == 'ResNet':
        re_batch_size = max(re_batch_size // 2, 1)
    if re_batch_size > len(oimages):
        re_batch_size = len(oimages)

    handles = []
    if model_type == 'ResNet':
        children_modules = list(list(children[Troj_Layer].children())[-1].children())
        print(len(children_modules), children_modules)
        bn3_module = children_modules[4]
        handle = bn3_module.register_forward_hook(get_after_bn3())
        handles.append(handle)
        if len(children_modules) > 5:
            iden_module = children_modules[-1]
            handle = iden_module.register_forward_hook(get_after_iden())
            handles.append(handle)
        else:
            iden_module = children_modules[0]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)
    elif model_type == 'Inception3':
        children_modules = []
        for j in range(len(list(children[Troj_Layer].modules()))):
            tm = list(children[Troj_Layer].modules())[j]
            if len(list(tm.children())) == 0:
                children_modules.append(tm)
        for j in range(len(children_modules)):
            print(j, children_modules[j])
        if children[Troj_Layer].__class__.__name__ == 'InceptionA':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[5]
            tmodule3 = children_modules[11]
            tmodule4 = children_modules[13]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionB':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[7]
            tmodule3 = children_modules[0]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_before_block())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionC':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[7]
            tmodule3 = children_modules[17]
            tmodule4 = children_modules[19]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionD':
            tmodule1 = children_modules[3]
            tmodule2 = children_modules[11]
            tmodule3 = children_modules[0]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_before_block())
            handles.append(handle)
        elif children[Troj_Layer].__class__.__name__ == 'InceptionE':
            tmodule1 = children_modules[1]
            tmodule2 = children_modules[5]
            tmodule3 = children_modules[7]
            tmodule4 = children_modules[13]
            tmodule5 = children_modules[15]
            tmodule6 = children_modules[17]
            handle = tmodule1.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule2.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule3.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule4.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule5.register_forward_hook(get_after_bns())
            handles.append(handle)
            handle = tmodule6.register_forward_hook(get_after_bns())
            handles.append(handle)
    elif model.__class__.__name__ == 'DenseNet':
        target_module = list(children[Troj_Layer].modules())[-1]
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model.__class__.__name__ == 'Net':
        target_module = list(children[Troj_Layer].modules())[-1]
        #print(target_module)
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model.__class__.__name__ == 'VGG':
        target_module = list(children[Troj_Layer].modules())[-1]
        #print(target_module)
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model.__class__.__name__ == 'LeNet5':
        target_module = list(children[Troj_Layer].modules())[-1]
        #print(target_module)
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model.__class__.__name__ == 'AlexNet':
        target_module = list(children[Troj_Layer].modules())[-1]
        #print(target_module)
        handle = target_module.register_forward_hook(get_after_bns())
        handles.append(handle)
    print('Target Layer', Troj_Layer, children[Troj_Layer], 'Neuron', Troj_Neuron, 'Target Label', Troj_Label)

    # delta = torch.randn(1,3,h,w).cuda()
    delta = torch.rand(1,3,h,w).cuda() * 2 - 1
    print(torch.rand(1))
    print(torch.rand(1))
    print(delta)
    mask = filter_img().reshape((1,1,h,w)) * 6 - 4
    mask= torch.FloatTensor(mask).cuda()
    delta.requires_grad = True
    mask.requires_grad = True
    optimizer = torch.optim.Adam([delta, mask], lr=re_mask_lr)
    print('before optimizing',)
    for e in range(re_epochs):
        print("reverse_engineer e:",e)
        facc = 0
        flogits = []
        p = np.random.permutation(oimages.shape[0])
        images = oimages[p]
        labels = olabels[p]
        #print(len(images)//re_batch_size)
        for i in range(len(images)//re_batch_size):
            optimizer.zero_grad()
            model.zero_grad()
            after_bn3.clear()
            before_block.clear()
            after_iden.clear()
            after_bns.clear()

            batch_data = torch.FloatTensor(images[re_batch_size*i:re_batch_size*(i+1)])
            batch_data = batch_data.cuda()

            con_mask = torch.tanh(mask)/2.0 + 0.5
            # delta is bounded by 0 and 1
            # use_delta = torch.clamp(delta, min=0.0 , max=1.0)
            # if e < try_epochs:
            #     use_delta = torch.tanh(torch.clamp(delta, -1, 1))/2.0 + 0.5
            # else:
            #     use_delta = torch.tanh(delta)/2.0 + 0.5
            use_delta = torch.tanh(delta)/2.0 + 0.5
            use_mask = con_mask
            in_data = use_mask * use_delta + (1-use_mask) * batch_data
            
            if model_type == 'ResNet':

                logits = model(in_data)
                logits_np = logits.cpu().detach().numpy()
                    
                after_bn3_t = torch.stack(after_bn3, 0)
                iden = None
                if len(before_block) > 0:
                    iden = before_block[0]
                else:
                    after_iden_t = torch.stack(after_iden, 0)
                    iden = after_iden_t
                inner_outputs_b = iden + after_bn3_t
                inner_outputs_a = F.relu(inner_outputs_b)

            elif model_type == 'Inception3':

                logits = model(in_data)
                logits_np = logits.cpu().detach().numpy()

                if children[Troj_Layer].__class__.__name__ == 'InceptionA':
                    after_bn1_t = torch.stack(after_bns[0*re_batch_size:1*re_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*re_batch_size:2*re_batch_size], 0)
                    after_bn3_t = torch.stack(after_bns[2*re_batch_size:3*re_batch_size], 0)
                    after_bn4_t = torch.stack(after_bns[3*re_batch_size:4*re_batch_size], 0)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t], 1)
                if children[Troj_Layer].__class__.__name__ == 'InceptionB':
                    after_bn1_t = torch.stack(after_bns[0*re_batch_size:1*re_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*re_batch_size:2*re_batch_size], 0)
                    before_in_t = before_block[0]
                    branch_pool = F.max_pool2d(before_in_t, kernel_size=3, stride=2)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, branch_pool], 1)
                if children[Troj_Layer].__class__.__name__ == 'InceptionC':
                    after_bn1_t = torch.stack(after_bns[0*re_batch_size:1*re_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*re_batch_size:2*re_batch_size], 0)
                    after_bn3_t = torch.stack(after_bns[2*re_batch_size:3*re_batch_size], 0)
                    after_bn4_t = torch.stack(after_bns[3*re_batch_size:4*re_batch_size], 0)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t], 1)
                if children[Troj_Layer].__class__.__name__ == 'InceptionD':
                    after_bn1_t = torch.stack(after_bns[0*re_batch_size:1*re_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*re_batch_size:2*re_batch_size], 0)
                    before_in_t = before_block[0]
                    branch_pool = F.max_pool2d(before_in_t, kernel_size=3, stride=2)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, branch_pool], 1)
                if children[Troj_Layer].__class__.__name__ == 'InceptionE':
                    after_bn1_t = torch.stack(after_bns[0*re_batch_size:1*re_batch_size], 0)
                    after_bn2_t = torch.stack(after_bns[1*re_batch_size:2*re_batch_size], 0)
                    after_bn3_t = torch.stack(after_bns[2*re_batch_size:3*re_batch_size], 0)
                    after_bn4_t = torch.stack(after_bns[3*re_batch_size:4*re_batch_size], 0)
                    after_bn5_t = torch.stack(after_bns[4*re_batch_size:5*re_batch_size], 0)
                    after_bn6_t = torch.stack(after_bns[5*re_batch_size:6*re_batch_size], 0)
                    inner_outputs_b = torch.cat([after_bn1_t, after_bn2_t, after_bn3_t, after_bn4_t, after_bn5_t, after_bn6_t], 1)

                inner_outputs_a = F.relu(inner_outputs_b)

            elif model_type == 'DenseNet':

                logits = model(in_data)
                logits_np = logits.cpu().detach().numpy()
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type == 'Net':
            
                logits = model(in_data)
                #print(F.softmax(logits))
                #print(F.softmax(logits).max(1))
                logits_np = logits.cpu().detach().numpy()
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type == 'VGG':
            
                logits = model(in_data)
                #print(F.softmax(logits))
                #print(F.softmax(logits).max(1))
                logits_np = logits.cpu().detach().numpy()
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)

            elif model_type == 'LeNet5':
            
                logits = model(in_data)
                #print(F.softmax(logits))
                #print(F.softmax(logits).max(1))
                logits_np = logits.cpu().detach().numpy()
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)
            
            elif model_type == 'AlexNet':
            
                logits = model(in_data)
                #print(F.softmax(logits))
                #print(F.softmax(logits).max(1))
                logits_np = logits.cpu().detach().numpy()
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)

            flogits.append(logits_np)
            loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, mask_loss, mask_nz, mask_add_loss, logits_loss\
                    = loss_fn(inner_outputs_b, inner_outputs_a, logits, use_mask, Troj_Neuron, int(Troj_Label), facc, e, re_epochs)
            if e > 0:
                loss.backward(retain_graph=True)
                optimizer.step()
                print(loss)
            # break
        flogits = np.concatenate(flogits, axis=0)
        preds = np.argmax(flogits, axis=1)

        # do not change Troj_Label
        Troj_Label2 = np.argmax(np.bincount(preds))

        facc = np.sum(preds == Troj_Label2) / float(preds.shape[0])

        '''if e % 10 == 0:
            print(e, 'loss', loss.cpu().detach().numpy(), 'acc {:.4f}'.format(facc),'target label', int(Troj_Label), int(Troj_Label2), 'logits_loss', logits_loss.cpu().detach().numpy(),\
                    'vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                    'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                    'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                    'mask_loss', mask_loss.cpu().detach().numpy(), 'mask_nz', mask_nz.cpu().detach().numpy(), 'mask_add_loss', mask_add_loss.cpu().detach().numpy())
            print('labels', flogits[:5,:])
            print('logits', np.argmax(flogits, axis=1))
            print('delta', use_delta[0,0,:5,:5])
            print('mask', use_mask[0,0,:5,:5])'''

        if facc > 0.99:
            break
    delta = use_delta.cpu().detach().numpy()
    con_mask = use_mask.cpu().detach().numpy()
    adv = in_data.cpu().detach().numpy()
    adv = deprocess(adv)

    #os.system('nvidia-smi')
    # cleaning up
    for handle in handles:
        handle.remove()

    return facc, adv, delta, con_mask, Troj_Label2

def re_mask(model_type, model, neuron_dict, children, images, labels, scratch_dirpath, re_epochs):
    validated_results = []
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, samp_label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_Layer = int(Troj_Layer.split('_')[1])

            print(weights_file)
            RE_img = os.path.join(scratch_dirpath,'imgs', '{0}_model_{1}_{2}_{3}_{4}.png'.format(    weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, Troj_size, samp_label))
            RE_mask = os.path.join(scratch_dirpath,'masks', '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(  weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, Troj_size, samp_label))
            RE_delta = os.path.join(scratch_dirpath,'deltas', '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, Troj_size, samp_label))
            
            max_acc = 0
            max_results = []
            for i  in range(mask_multi_start):
                acc, rimg, rdelta, rmask, optz_label = reverse_engineer(model_type, model, children, images, labels, weights_file, Troj_Layer, Troj_Neuron, samp_label, Troj_size, re_epochs)

                #print(rimg.shape)
                re_save_path = os.path.join(scratch_dirpath,'imgs', '{0}_model_{1}_{2}_{3}_{4}_{5}_{6}.png'.format(    weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, Troj_size, samp_label, optz_label, acc))
                skimage.io.imsave(re_save_path, rimg[0])
                
                #skimage.io.imsave(RE_img, rimg[0])
                #skimage.io.imsave(RE_mask, rmask.transpose(0,2,3,1)[0])
                #print(rdelta.shape)
                #skimage.io.imsave(RE_delta, rdelta.transpose(0,2,3,1)[0])
                # clear cache
                torch.cuda.empty_cache()

                if Print_Level > 0:
                    print('RE mask', Troj_Layer, Troj_Neuron, 'Label', optz_label, 'RE acc', acc)
                if acc > max_acc:
                    max_acc = acc
                    max_results = (rimg, rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, acc)
            if max_acc >= reasr_bound - 0.2:
                validated_results.append( max_results )
            if max_acc > 0.99 and optz_label == samp_label:
                break
        return validated_results


def stamp(n_img, delta, mask):
    mask0 = nc_filter_img()
    mask = mask * mask0
    r_img = n_img.copy()
    mask = mask.reshape((1,1,h,w))
    r_img = n_img * (1-mask) + delta * mask
    #print("stamp\n\n\n")
    #print(delta)
    #print(n_img)
    return r_img

def filter_stamp(n_img, trigger):
    if channel_last:
        t_image = tf.placeholder(tf.float32, shape=(None, h, w, 3))
        ti_image = t_image
    else:
        t_image = tf.placeholder(tf.float32, shape=(None, 3, h, w))
        ti_image = tf.transpose(t_image, [0,2,3,1])

    tdelta = tf.placeholder(tf.float32, shape=(12, 3))
    imax =  tf.nn.max_pool( ti_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    imin = -tf.nn.max_pool(-ti_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    iavg =  tf.nn.avg_pool( ti_image, ksize=[1,window_size,window_size,1], strides=[1,1,1,1], padding='SAME')
    i_image = tf.reshape( tf.matmul( tf.reshape( tf.concat([ti_image, imax, imin, iavg], axis=3), (-1,12)) , tdelta), [-1,h,w,3])

    if not channel_last:
        i_image = tf.transpose(i_image, [0,3,1,2])

    with tf.Session() as sess:
        r_img = sess.run(i_image, {t_image: n_img, tdelta:trigger})
    return r_img

def test(model, model_type, test_xs, test_ys, result, scratch_dirpath, test_index, log_dir, log_name, mode='mask'):
    
    re_batch_size = config['re_batch_size']
    if model_type == 'DenseNet':
        re_batch_size = max(re_batch_size // 3, 1)
    if model_type == 'ResNet':
        re_batch_size = max(re_batch_size // 2, 1)
    if re_batch_size > len(test_xs):
        re_batch_size = len(test_xs)

    clean_images = test_xs

    # if mode == 'mask':
    if True:
        rimg, rdelta, rmask, tlabel = result[:4]
        rmask = rmask * rmask > mask_epsilon
        t_images = stamp(clean_images, rdelta, rmask)

    saved_images = deprocess(t_images)

    rt_images = t_images
    
    if Print_Level > 0:
        print(np.amin(rt_images), np.amax(rt_images))
    
    yt = np.zeros(len(rt_images)).astype(np.int32) + tlabel
    fpreds = []
    for i in range(len(rt_images)//re_batch_size):
        batch_data = torch.FloatTensor(rt_images[re_batch_size*i:re_batch_size*(i+1)])
        batch_data = batch_data.cuda()
        preds = model(batch_data)
        fpreds.append(preds.cpu().detach().numpy())
    fpreds = np.concatenate(fpreds)

    preds = np.argmax(fpreds, axis=1)
    #print(preds)
    #print(test_ys)
    
    if score_type == "mean":
        print("tlabel:",tlabel)
        for i in range(num_classes):
            print("class:",i)
            #print(preds)
            #print(test_ys)
            class_i_index = np.where(test_ys==i)
            pred_class_i = preds[class_i_index]
            score = float(np.sum(tlabel == pred_class_i))/float(pred_class_i.shape[0])
            print(score)
        score = float(np.sum(tlabel == preds))/float(yt.shape[0])
        
    if score_type == "mean_exclude_target":
        print("tlabel:",tlabel)
        correct_num = 0
        total_num = 0
        for i in range(num_classes):
            print("class:",i)
            print(len(preds))
            print(len(test_ys))
            class_i_index = np.where(test_ys==i)
            print(class_i_index)
            pred_class_i = preds[class_i_index]
            score = float(np.sum(tlabel == pred_class_i))/float(pred_class_i.shape[0])
            if i != tlabel:
                correct_num = correct_num + float(np.sum(tlabel == pred_class_i))
                total_num = total_num + float(pred_class_i.shape[0])
            
            print(score)
            
        score = float(correct_num)/float(total_num)
        #score = float(np.sum(tlabel == preds))/float(yt.shape[0])

    elif score_type == "max":
        print("tlabel:",tlabel)
        score_list = []
        for i in range(num_classes):
            print("class:",i)
            #print(preds)
            #print(test_ys)
            class_i_index = np.where(test_ys==i)
            pred_class_i = preds[class_i_index]
            score = float(np.sum(tlabel == pred_class_i))/float(pred_class_i.shape[0])
            print(score)
            if i != tlabel:
                score_list.append(score)
        
        #score = float(np.sum(tlabel == preds))/float(yt.shape[0])
        score = max(score_list)
    
    
    
    #print('label', tlabel, 'score', score)
    with open( log_dir+log_name ,'a') as f:
        f.write('index:' + str(test_index)+' label:'+str(tlabel)+' score:'+str(score))
        f.write('\n')
    test_result = str(tlabel)+"_"+str(score)
    
    return score,test_result

def run_abs(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, log_dir, log_name, arch_local="nin", example_img_format='png', troj_size_override=None, num_classes_override = None, total_size_override = None, re_bs_override = None, dpsgd_convert=0 ):
    start = time.time()
    
    seed_torch(333)
    
    print(torch.rand(1))
    print(torch.rand(1))
    
    global arch
    arch = arch_local
    
    if troj_size_override:
        global Troj_size
        Troj_size = troj_size_override
        
    if num_classes_override:
        global num_classes
        num_classes = num_classes_override
    
    if total_size_override:
        global w
        global h
        w = total_size_override
        h = total_size_override
    
    if re_bs_override:
        global config
        config['re_batch_size'] = re_bs_override
            
    

    # create dirs
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'imgs')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'masks')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'temps')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'deltas')))

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    #random.shuffle(fns)
    imgs = []
    fys = []
    image_mins = []
    image_maxs = []
    #print(fns)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    for fn in fns:
        # read the image (using skimage)
        img = skimage.io.imread(fn)
        fys.append(int(fn.split('_')[-3]))
        # perform tensor formatting and normalization explicitly
        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))

        img = img/255

        for i in range(len(mean)):
            img[i] = (img[i] - mean[i])/std[i]

        img = np.expand_dims(img, 0)
        
        image_mins.append(np.min(img))
        image_maxs.append(np.max(img))
        imgs.append(img)
    fxs = np.concatenate(imgs)
    fys = np.array(fys)
    image_min = np.mean(image_mins)
    image_max = np.mean(image_maxs)
    
    print('number of seed images', len(fys), fys.shape)

    test_xs = fxs
    test_ys = fys
    
    n_sample_imgs = 200
    sample_xs = []
    sample_ys = []
    sample_slots = np.zeros(num_classes)
    for i in range(len(fys)):
        #print(fys[i])
        if sample_slots[fys[i]] < n_sample_imgs//num_classes:
            sample_xs.append(fxs[i])
            sample_ys.append(fys[i])
            sample_slots[fys[i]] += 1
        if np.sum(sample_slots) >= n_sample_imgs:
            break
    sample_xs = np.array(sample_xs)
    sample_ys = np.array(sample_ys)

    xs = sample_xs.copy()
    ys = sample_ys.copy()
    optz_xs = []
    optz_ys = []
    optz_slots = np.zeros(num_classes)
    for i in range(len(fys)):
        if optz_slots[fys[i]] < n_re_samples//num_classes:
            optz_xs.append(fxs[i])
            optz_ys.append(fys[i])
            optz_slots[fys[i]] += 1
        if np.sum(optz_slots) >= n_re_samples:
            break
    optz_xs = np.array(optz_xs)
    optz_ys = np.array(optz_ys)
    xs = optz_xs
    ys = optz_ys

    if Print_Level > 0:
        print('# samples for RE', len(ys))

    if arch == "nin":
        model = Net()
    elif arch == "vgg16":
        model = vgg.vgg16(num_classes=num_classes)
    elif arch == "vgg11":
        model = vgg.vgg11(num_classes=num_classes)
    elif arch == "vgg11bn":
        model = vgg_trojai.vgg11_bn(num_classes=num_classes)
    elif arch == "resnet18":
        model = resnet.resnet18(num_classes=num_classes)

    try:
        #print(1)
        model.load_state_dict(torch.load(model_filepath))
        #model.load_state_dict(torch.load(model_filepath),strict=True)
    except:
        model = torch.load(model_filepath)
    #print(model)
    
    
    
    model = model.cuda()
    
    model.eval()
    correct = 0
    for i in range(len(test_xs)//10):
        batch_data = torch.FloatTensor(test_xs[10*i:10*(i+1)])
        batch_data = batch_data.cuda()

        preds = model(batch_data)

        labels = torch.Tensor(test_ys[10*i:10*(i+1)]).cuda()
        labels = labels.view(10,-1)
        
        preds = preds.data.max(1, keepdim=True)[1]

        correct += preds.eq(labels).cpu().sum()

    print(correct)
    print(len(test_xs))
    
    target_layers = []
    model_type = model.__class__.__name__
    if model_type == 'ResNet':
        children = list(model.children())

        children.insert(-1, torch.nn.Flatten())
        #print(children)
        # target_layers = ['Bottleneck']
        target_layers = ['Sequential']
        #target_layers = ['AvgPool2d']
    elif model_type == 'Inception3':
        children = list(model.children())
        children.insert(3, torch.nn.MaxPool2d(kernel_size=3, stride=2))
        children.insert(6, torch.nn.MaxPool2d(kernel_size=3, stride=2))
        children.insert(-1, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['InceptionA', 'InceptionB', 'InceptionC', 'InceptionD', 'InceptionE']
    elif model_type == 'DenseNet':
        children = list(model.children())
        children = list(children[0].children()) + children[1:]
        nchildren = []
        for c in children:
            if c.__class__.__name__ == '_Transition':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-1, torch.nn.ReLU(inplace=True))
        children.insert(-1, torch.nn.AdaptiveAvgPool2d((1, 1)))
        children.insert(-1, torch.nn.Flatten())
        print(children)
        target_layers = ['BatchNorm2d']
    elif model_type == 'Net':
        children = list(model.children())
        children = list(children[0].children()) + children[1:]

        children.append(torch.nn.Flatten())
        print(children)
        target_layers = ['Conv2d']
    elif model_type == 'VGG':
        children = list(model.children())
        children = list(children[0].children()) + children[1:]

        children.insert(-1, torch.nn.Flatten())
        print(children)
        target_layers = ['Conv2d']
        
    elif model_type == 'LeNet5':
        children = list(model.children())

        children.insert(-3, torch.nn.Flatten())
        print(children)
        target_layers = ['Conv2d']
    elif model_type == 'AlexNet':
        #children = list(model.children())
        children = list(model.children())
        children = list(children[0].children()) + list(children[1].children())

        children.insert(-7, torch.nn.Flatten())
        print(children)
        target_layers = ['Conv2d']
    else:
        print('other model', model_type)
        sys.exit()

    if Print_Level > 0:
        print('layers')
        for i in range(len(children)):
            print(i, children[i], type(children[i]))

    if Print_Level > 0:
        print('image range', np.amin(test_xs), np.amax(test_xs))

    neuron_dict = {}
    sampling_val = 0

    maxes =  check_values( sample_xs, sample_ys, model, children, target_layers)
    torch.cuda.empty_cache()
    all_ps, sample_layers = sample_neuron(sample_xs, sample_ys, model, children, target_layers, model_type, maxes)
    torch.cuda.empty_cache()

    neuron_dict, sampling_val = read_all_ps(model_filepath, all_ps, sample_layers, top_k = top_n_neurons)
    print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', neuron_dict)
    
    with open( log_dir+log_name ,'a') as f:
        f.write('Compromised Neuron Candidates (Layer, Neuron, Target_Label):'+ str(neuron_dict))
        f.write('\n')

    sample_end = time.time()

    results = re_mask(model_type, model, neuron_dict, children, xs, ys, scratch_dirpath, re_epochs)
    reasr_info = []
    reasrs = []
    
    test_index = 0
    if len(results) > 0:
        reasrs = []
        test_results = []
        for result in results:
            reasr,test_result = test(model, model_type, test_xs, test_ys, result, scratch_dirpath, test_index, log_dir, log_name)
            reasrs.append(reasr)
            test_results.append(test_result)
            adv, rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, acc = result
            rmask = rmask * rmask > mask_epsilon
            reasr_info.append([reasr, 'mask', str(optz_label), str(samp_label), RE_img, RE_mask, RE_delta, np.sum(rmask), acc])
            test_index = test_index + 1
        print(str(model_filepath), 'mask check', max(reasrs))
    else:
        print(str(model_filepath), 'mask check', 0)

    optm_end = time.time()
    if len(reasrs) > 0:
        freasr = max(reasrs)
        f_id = reasrs.index(freasr)
    else:
        freasr = 0
        f_id = 0
    max_reasr = 0
    for i in range(len(reasr_info)):
        print('reasr info {0}'.format( ' '.join([str(_) for _ in reasr_info[i]]) ))
        reasr = reasr_info[i][0]
        if reasr > max_reasr :
            max_reasr = reasr
    print('{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(\
            model_filepath, model_type, 'mask', freasr, 'sampling val', sampling_val, 'time', sample_end - start, optm_end - sample_end,) )
    del model
    torch.cuda.empty_cache()
    return test_results