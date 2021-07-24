import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tensorflow import keras
from utils.tools import *
from prettytable import PrettyTable


class CIFAR10DATASET(Dataset):
    def __init__(self, X, Y, ids, transform=None):
        self.X = X
        self.Y = Y
        self.ids = ids
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.X[self.ids[item]], self.Y[self.ids[item]]
        assert x.shape == (32, 32, 3)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y

    def __len__(self):
        return len(self.ids)


def _get_cifar10_dataLoaders(X, Y, batch_size=10, transform=None, rand_set_all=None):
    if rand_set_all is None:
        rand_set_all = []
    dict_users = {i: np.array([], dtype='int64') for i in range(100)}

    idxs_dict = {}
    for i in range(len(X)):
        label = Y[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_users = 100
    num_classes = 10
    shard_per_client = 2
    shard_per_class = int(shard_per_client * num_users / num_classes)  # 2 shards per client * 100 clients / 10 classes
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x
    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(Y[value])
        assert len(x) <= shard_per_client
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(X)
    assert len(set(list(test))) == len(X)

    dataLoaders = {}
    clients = [i for i in range(100)]
    for user_id in clients:
        dataset = CIFAR10DATASET(X, Y, ids=dict_users[user_id], transform=transform)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        dataLoaders[user_id] = data_loader

    return clients, dataLoaders, rand_set_all


def get_cifar10_dataLoaders(batch_size=10, train_transform=None, test_transform=None):
    # todo tdye: Note that the random seed might affect the model initialization, so reset it latter
    setup_seed(24)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape((50000,))
    y_test = y_test.reshape((10000,))

    table = PrettyTable(['TrainingX.', 'TrainingY.', 'TestX.', 'TestY.'])
    table.add_row([x_train.shape, y_train.shape, x_test.shape, y_test.shape])
    print(table)

    train_all_clients, trainLoaders, rand_set_all = _get_cifar10_dataLoaders(X=x_train, Y=y_train,
                                                                             batch_size=batch_size,
                                                                             transform=train_transform, rand_set_all=[])
    test_all_clients, testLoaders, rand_set_all = _get_cifar10_dataLoaders(X=x_test, Y=y_test, batch_size=batch_size,
                                                                           transform=test_transform,
                                                                           rand_set_all=rand_set_all)
    train_all_clients.sort()
    test_all_clients.sort()
    assert train_all_clients == test_all_clients
    return train_all_clients, trainLoaders, testLoaders


if __name__ == '__main__':
    _clients, _trainLoaders, _testLoaders = get_cifar10_dataLoaders()
    for _client in _clients:
        ls = []
        for _, (data, labels) in enumerate(_trainLoaders[_client]):
            ls.extend(list(np.array(torch.unique(labels))))
        for _, (data, labels) in enumerate(_testLoaders[_client]):
            ls.extend(list(np.array(torch.unique(labels))))
        print(np.unique(np.array(ls)))
