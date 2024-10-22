import os
import time
import random
import skimage.io
import numpy as np
import scipy.io
import torch, random, time
from torch.utils.data import Dataset
import torchvision as tv
from torchvision.datasets import CIFAR100
from hrrp3 import hrrp3_Dataset, gaf12_Dataset
from trainer import tensor2cuda

def count_data(data_dict):
    num = 0
    for key in data_dict.keys():
        num += len(data_dict[key])
    return num

class self_DataLoader(Dataset):
    def __init__(self, root, train=True, dataset='hrrp3', seed=1, nway=5):
        super(self_DataLoader, self).__init__()

        self.seed = seed
        self.nway = nway
        self.num_labels = 12
        self.input_channels = 1
        self.size = 32

        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5071, 0.4866, 0.4409],
                [0.2673, 0.2564, 0.2762])
            ])

        self.full_data_dict, self.few_data_dict = self.load_data(root, train, dataset)

        print('full_data_num: %d' % count_data(self.full_data_dict))
        print('few_data_num: %d' % count_data(self.few_data_dict))

    def load_data(self, root, train, dataset):
        if dataset == 'cifar100':
            few_selected_label = random.Random(self.seed).sample(range(self.num_labels), self.nway)
            print('selected labeled', few_selected_label)

            full_data_dict = {}
            few_data_dict = {}

            d = CIFAR100(root, train=train, download=True)

            for i, (data, label) in enumerate(d):

                data = self.transform(data)

                if label in few_selected_label:
                    data_dict = few_data_dict
                else:
                    data_dict = full_data_dict

                if label not in data_dict:
                    data_dict[label] = [data]
                else:
                    data_dict[label].append(data)

                print(i + 1)

            return full_data_dict, few_data_dict

        if dataset == 'hrrp3':
            few_selected_label = random.Random(self.seed).sample(range(self.num_labels), self.nway)
            print('selected labeled', few_selected_label)

            full_data_dict = {}
            few_data_dict = {}

            data_dir = 'data/hrrp3'  # Replace with the path to your HRRP3 data
            transform = None  # Define your transform if necessary
            d = hrrp3_Dataset(root=data_dir, train=True, transform=transform)

            for i, (data, label) in enumerate(d):

                if label in few_selected_label:
                    data_dict = few_data_dict
                else:
                    data_dict = full_data_dict

                if label not in data_dict:
                    data_dict[label] = [data]
                else:
                    data_dict[label].append(data)

                # print(i + 1)

            return full_data_dict, few_data_dict

        if dataset == 'gaf12':
            few_selected_label = random.Random(self.seed).sample(range(self.num_labels), self.nway)
            print('selected labeled', few_selected_label)

            full_data_dict = {}
            few_data_dict = {}

            data_dir = 'data/gaf12'  # Replace with the path to your GAF12 data
            transform = None  # Define your transform if necessary
            d = gaf12_Dataset(root=data_dir, train=True, transform=transform)

            for i, (data, label) in enumerate(d):

                if label in few_selected_label:
                    data_dict = few_data_dict
                else:
                    data_dict = full_data_dict

                if label not in data_dict:
                    data_dict[label] = [data]
                else:
                    data_dict[label].append(data)

                # print(i + 1)

            return full_data_dict, few_data_dict

    def load_batch_data(self, train=True, batch_size=16, nway=5, num_shots=10):
        if train:
            data_dict = self.full_data_dict
        else:
            data_dict = self.few_data_dict

        x = []
        label_y = [] # fake label: from 0 to (nway - 1)
        one_hot_y = [] # one hot for fake label
        class_y = [] # real label

        xi = []
        label_yi = []
        one_hot_yi = []


        map_label2class = []

        ### the format of x, label_y, one_hot_y, class_y is
        ### [tensor, tensor, ..., tensor] len(label_y) = batch size
        ### the first dimension of tensor = num_shots

        for i in range(batch_size):

            # sample the class to train
            sampled_classes = random.sample(data_dict.keys(), nway)

            positive_class = random.randint(0, nway - 1)

            label2class = torch.LongTensor(nway)

            single_xi = []
            single_one_hot_yi = []
            single_label_yi = []
            single_class_yi = []


            for j, _class in enumerate(sampled_classes):
                if j == positive_class:
                    ### without loss of generality, we assume the 0th
                    ### sampled  class is the target class
                    sampled_data = random.sample(data_dict[_class], num_shots+1)

                    x.append(sampled_data[0])
                    label_y.append(torch.LongTensor([j]))

                    one_hot = torch.zeros(nway)
                    one_hot[j] = 1.0
                    one_hot_y.append(one_hot)

                    class_y.append(torch.LongTensor([_class]))

                    shots_data = sampled_data[1:]
                else:
                    shots_data = random.sample(data_dict[_class], num_shots)

                single_xi += shots_data
                single_label_yi.append(torch.LongTensor([j]).repeat(num_shots))
                one_hot = torch.zeros(nway)
                one_hot[j] = 1.0
                single_one_hot_yi.append(one_hot.repeat(num_shots, 1))

                label2class[j] = _class

            shuffle_index = torch.randperm(num_shots*nway)
            xi.append(torch.stack(single_xi, dim=0)[shuffle_index])
            label_yi.append(torch.cat(single_label_yi, dim=0)[shuffle_index])
            one_hot_yi.append(torch.cat(single_one_hot_yi, dim=0)[shuffle_index])

            map_label2class.append(label2class)

        return [torch.stack(x, 0), torch.cat(label_y, 0), torch.stack(one_hot_y, 0), \
            torch.cat(class_y, 0), torch.stack(xi, 0), torch.stack(label_yi, 0), \
            torch.stack(one_hot_yi, 0), torch.stack(map_label2class, 0)]

    def load_tr_batch(self, batch_size=16, nway=5, num_shots=10):
        # 根据需要修改此方法以生成支持集和查询集
        d = self.load_batch_data(True, batch_size, nway, num_shots)
        query_data = [tensor[:1] for tensor in d]
        support_data = [tensor[1:] for tensor in d]
        support_data_cuda = [tensor2cuda(data) for data in support_data]
        query_data_cuda = [tensor2cuda(data) for data in query_data]
        return support_data_cuda, query_data_cuda

    def load_te_batch(self, batch_size=16, nway=5, num_shots=10):
        return self.load_batch_data(False, batch_size, nway, num_shots)

    def get_data_list(self, data_dict):
        data_list = []
        label_list = []
        for i in data_dict.keys():
            for data in data_dict[i]:
                data_list.append(data)
                label_list.append(i)

        now_time = time.time()
        now_time = time.time()

        random.Random(now_time).shuffle(data_list)
        random.Random(now_time).shuffle(label_list)

        return data_list, label_list

    def get_full_data_list(self):
        return self.get_data_list(self.full_data_dict)

    def get_few_data_list(self):
        return self.get_data_list(self.few_data_dict)

if __name__ == '__main__':
    D = self_DataLoader('data', True)

    [x, label_y, one_hot_y, class_y, xi, label_yi, one_hot_yi, class_yi] = \
        D.load_tr_batch(batch_size=16, nway=5, num_shots=10)
    print(x.size(), label_y.size(), one_hot_y.size(), class_y.size())
    print(xi.size(), label_yi.size(), one_hot_yi.size(), class_yi.size())

    # print(label_y)
    # print(one_hot_y)

    print(label_yi[0])
    print(one_hot_yi[0])