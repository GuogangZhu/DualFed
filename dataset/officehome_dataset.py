import os
from PIL import Image
import pandas as pd

def load_officehome(base_dir, domain, train_num=105, test_num=-1):
    # load image paths and lables for *.pkl file
    data_path = os.path.join(base_dir, domain)

    train_df = pd.read_csv(os.path.join(base_dir, '{}_train.csv'.format(domain)))
    test_df = pd.read_csv(os.path.join(base_dir, '{}_test.csv'.format(domain)))

    train_paths = train_df['train_path'].values
    test_paths = test_df['test_path'].values

    train_labels = train_df['train_label'].values
    test_labels = test_df['test_label'].values

    train_imgs = []
    test_imgs = []

    # trunctate the dataset
    if train_num <= len(train_paths):
        train_paths = train_paths[:train_num]
        train_labels = train_labels[:train_num]

    if test_num <= len(test_paths):
        test_paths = test_paths[:test_num]
        test_labels = test_labels[:test_num]

    # load images in train dataset
    for i in range(len(train_paths)):
        img_path = os.path.join(data_path, train_paths[i])
        img = Image.open(img_path)
        train_imgs.append(img.copy())
        img.close()

    for i in range(len(test_paths)):
        img_path = os.path.join(data_path, test_paths[i])
        img = Image.open(img_path)
        test_imgs.append(img.copy())
        img.close()

    print('Load {} Dataset...'.format(domain))
    print('Train Dataset Size:', len(train_imgs))
    print('Test Dataset Size:', len(test_imgs))

    return train_imgs, train_labels, test_imgs, test_labels


