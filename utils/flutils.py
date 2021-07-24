import torch
from csvec import CSVec
from torchvision.transforms import transforms

# datasets
from data.cifar10.cifar10 import get_cifar10_dataLoaders
from data.cifar10.cifar10_iid import get_cifar10_dataLoaders as get_cifar10_iid_dataLoaders

# models
from models import *


def setup_datasets(dataset, batch_size):
    users, trainLoaders, testLoaders = [], [], []
    if dataset == 'cifar10':
        trainTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        testTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        users, trainLoaders, testLoaders = get_cifar10_dataLoaders(batch_size=batch_size,
                                                                   train_transform=trainTransform,
                                                                   test_transform=testTransform)
    elif dataset == 'cifar10_iid':
        trainTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        testTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        users, trainLoaders, testLoaders = get_cifar10_iid_dataLoaders(batch_size=batch_size,
                                                                       train_transform=trainTransform,
                                                                       test_transform=testTransform)
    return users, trainLoaders, testLoaders


def select_model(algorithm, model_name):
    model = None
    if algorithm == 'fetchsgd':
        if model_name == 'cifar10':
            model = FetchSGD_CIFAR10()
        elif model_name == 'resnet9':
            model = ResNet9(num_classes=10)
        else:
            print(f"Unimplemented Model {model_name}")
    else:
        print(f"Unimplemented Algorithm {algorithm}")
    return model


def fedAverage(updates):
    total_weight = 0
    (clientSamplesNum, new_params) = updates[0]

    for (clientSamplesNum, client_params) in updates:
        total_weight += clientSamplesNum

    for k in new_params.keys():
        for i in range(0, len(updates)):
            client_samples, client_params = updates[i]
            # weight
            w = client_samples / total_weight
            if i == 0:
                new_params[k] = client_params[k] * w
            else:
                new_params[k] += client_params[k] * w
    # return global model params
    return new_params


def avgMetric(metricList):
    total_weight = 0
    total_metric = 0
    for (samplesNum, metric) in metricList:
        total_weight += samplesNum
        total_metric += samplesNum * metric
    average = total_metric / total_weight

    return average


def get_param_vec(model):
    param_vec = []
    for p in model.parameters():
        if p.requires_grad:
            param_vec.append(p.data.view(-1).float())
    return torch.cat(param_vec)


def set_param_vec(model, param_vec):
    start = 0
    for p in model.parameters():
        if p.requires_grad:
            end = start + p.numel()
            p.data.zero_()
            p.data.add_(param_vec[start:end].view(p.size()))
            start = end


def args2sketch(grad_size=918090, num_cols=100000, num_rows=5, device=torch.device('cuda:0'), num_blocks=10):
    return CSVec(d=grad_size, c=num_cols, r=num_rows, device=device, numBlocks=num_blocks)


def clip_grad(l2_norm_clip, record):
    try:
        l2_norm = torch.norm(record)
    except:
        l2_norm = record.l2estimate()
    if l2_norm < l2_norm_clip:
        return record
    else:
        return record / float(torch.abs(torch.tensor(l2_norm) / l2_norm_clip))


def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    # on a gpu, sorting is faster than pytorch's topk method
    # topkIndices = torch.sort(vec**2)[1][-k:]
    # however, torch.topk is more space efficient

    # topk on cuda returns what looks like uninitialized memory if
    # vals has nan values in it
    # saving to a zero-initialized output array instead of using the
    # output of topk appears to solve this problem
    topkVals = torch.zeros(k, device=vec.device)
    topkIndices = torch.zeros(k, device=vec.device).long()
    torch.topk(vec ** 2, k, sorted=False, out=(topkVals, topkIndices))

    ret = torch.zeros_like(vec)
    if len(vec.size()) == 1:
        ret[topkIndices] = vec[topkIndices]
    elif len(vec.size()) == 2:
        rows = torch.arange(vec.size()[0]).view(-1, 1)
        ret[rows, topkIndices] = vec[rows, topkIndices]
    return ret


if __name__ == '__main__':
    sketch = args2sketch(grad_size=10000, num_cols=1000, num_rows=5, device=torch.device('cuda:0'), num_blocks=10)
    print(sketch)
    x = torch.rand(10000, device=torch.device("cuda:0"))
    print(x.size())
    print(sketch.accumulateVec(x))
    y = sketch.unSketch(k=10)
    print(sum(abs(y - x)) / len(y - x))
