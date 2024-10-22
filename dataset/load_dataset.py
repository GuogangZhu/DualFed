from dataset.domainnet_dataset import load_domainnet
from dataset.pacs_dataset import load_pacs
from dataset.officehome_dataset import load_officehome
import torchvision.transforms as transforms
from dataset.datasets_ import DomainNetDataset, PACSDataset, OfficeHomeDataset, DatasetSplit
from torch.utils.data import DataLoader
import numpy as np

# get domainnet dataset
def domainnet_dataset_read(domains, base_dir=None, train_num=105, test_num=-1, scale=256):
    dataset_train = {}
    dataset_test = {}

    # train transform
    transform_train = transforms.Compose([
        transforms.Resize([scale, scale]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize([scale, scale]),
        transforms.ToTensor()
    ])

    for domain in domains:
        train_imgs, train_labels, test_imgs, test_labels = \
            load_domainnet(base_dir=base_dir, domain=domain, train_num=train_num, test_num=test_num)

        dataset_train[domain] = DomainNetDataset(train_imgs, train_labels, transform=transform_train)
        dataset_test[domain] = DomainNetDataset(test_imgs, test_labels, transform=transform_test)

    return dataset_train, dataset_test

# get pacs dataset
def pacs_dataset_read(domains, base_dir=None, train_num=105, test_num=-1, scale=256):
    dataset_train = {}
    dataset_test = {}

    # train transform
    transform_train = transforms.Compose([
        transforms.Resize([scale, scale]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize([scale, scale]),
        transforms.ToTensor()
    ])

    for domain in domains:
        train_imgs, train_labels, test_imgs, test_labels = \
            load_pacs(base_dir=base_dir, domain=domain, train_num=train_num, test_num=test_num)

        dataset_train[domain] = PACSDataset(train_imgs, train_labels, transform=transform_train)
        dataset_test[domain] = PACSDataset(test_imgs, test_labels, transform=transform_test)

    return dataset_train, dataset_test

# get officehome dataset
def officehome_dataset_read(domains, base_dir=None, train_num=105, test_num=-1, scale=256):
    dataset_train = {}
    dataset_test = {}

    # train transform
    transform_train = transforms.Compose([
        transforms.Resize([scale, scale]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize([scale, scale]),
        transforms.ToTensor()
    ])

    for domain in domains:
        train_imgs, train_labels, test_imgs, test_labels = \
            load_officehome(base_dir=base_dir, domain=domain, train_num=train_num, test_num=test_num)

        dataset_train[domain] = OfficeHomeDataset(train_imgs, train_labels, transform=transform_train)
        dataset_test[domain] = OfficeHomeDataset(test_imgs, test_labels, transform=transform_test)

    return dataset_train, dataset_test


def dataset_loader(datasets, bs, num_users, iid_sampling, shuffle=True):
    dataset_loader = {domain:[] for domain in datasets}
    dataset_len = {domain:[] for domain in datasets}
    for domain in datasets:
        #sampling
        if iid_sampling:
            dict_users = sampling_iid(datasets[domain], num_users)
        else:
            dict_users = sampling_iid(datasets[domain], num_users[domain])
        # generate dataset loader
        for idx in range(num_users):
            dataset_len[domain].append(len(dict_users[idx]))
            dataset_loader[domain].append(DataLoader(DatasetSplit(datasets[domain], dict_users[idx]),
                                                     batch_size=bs, shuffle=shuffle, num_workers=0))
    return dataset_loader, dataset_len

def sampling_iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users