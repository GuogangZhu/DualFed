import numpy as np
import os
from PIL import Image

def load_domainnet(base_dir, domain, train_num=105, test_num=-1):
    # load image paths and lables for *.pkl file
    train_paths, train_text_labels = np.load('{}DomainNet/split/{}_train.pkl'.format(base_dir, domain), allow_pickle=True)
    test_paths, test_text_labels = np.load('{}DomainNet/split/{}_test.pkl'.format(base_dir, domain), allow_pickle=True)

    label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}

    # transform text labels to digit labels
    train_labels = [label_dict[text] for text in train_text_labels]
    test_labels = [label_dict[text] for text in test_text_labels]

    train_imgs = []
    test_imgs = []

    # load images in train dataset
    for i in range(len(train_paths)):
        img_path = os.path.join(base_dir, train_paths[i])
        img = Image.open(img_path)
        train_imgs.append(img.copy())
        img.close()

    for i in range(len(test_paths)):
        img_path = os.path.join(base_dir, test_paths[i])
        img = Image.open(img_path)
        test_imgs.append(img.copy())
        img.close()

    if train_num <= len(train_imgs):
        train_imgs = train_imgs[:train_num]
        train_labels = train_labels[:train_num]

    if test_num <= len(test_imgs):
        test_imgs = test_imgs[:test_num]
        test_labels = test_labels[:test_num]

    print('Load {} Dataset...'.format(domain))
    print('Train Dataset Size:', len(train_imgs))
    print('Test Dataset Size:', len(test_imgs))

    return train_imgs, train_labels, test_imgs, test_labels