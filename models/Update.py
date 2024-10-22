#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from utils.MMD import MMDLoss
from utils.K_Moment import K_Moment
from utils.coral import CORAL_LOSS

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, dataset_loader_train=None, lr = 0.01, momentum = 0.5, local_ep = 5, device = 'GPU',
                 tranfer = True, transfer_lambda = 1, verbose = True):
        self.dataset_loader_train = dataset_loader_train
        # self.source_index = source_index
        # self.target_index = target_index

        self.lr = lr
        self.momentum = momentum
        self.local_ep = local_ep
        self.device = device

        self.transfer = tranfer
        self.tranfer_lambda = transfer_lambda
        self.verbose = verbose

        self.loss_cls_func = nn.CrossEntropyLoss()
        self.loss_mmd_func = MMDLoss()
        self.loss_moment_func = K_Moment()
        self.loss_coral_fun = CORAL_LOSS()

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum)

        epoch_loss = {'cls':[], 'dis':[], 'total':[]}
        for iter in range(self.local_ep):
            batch_loss = {'cls':[], 'dis':[], 'total':[]}
            for batch_idx, data in enumerate(self.dataset_loader_train):
                # img_t = autograd.Variable(data['target_imgs'].to(self.device)) #列表
                img_t = data['target_imgs']
                img_s = autograd.Variable(data['source_imgs'][0].to(self.device))
                label_s = autograd.Variable(data['source_labels'][0].long().to(self.device))
                # images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                source_feature, log_probs = net(img_s)
                target_feature = net(img_s)[0]

                for target_idx in range(len(img_t)):
                    target_feature += net(autograd.Variable(img_t[target_idx]).to(self.device))[0]

                target_feature /= torch.tensor(len(img_t) + 1, dtype=torch.float32)

                loss_cls = self.loss_cls_func(log_probs, label_s)

                # loss_dis = self.euclidean(source_feature, target_feature)
                loss_dis = torch.zeros(1)
                # self.loss_dis_func(source_feature, target_feature)
                # loss_dis = self.loss_cls_func(log_probs, label_s)
                if self.transfer:
                    # loss_dis = self.loss_mmd_func(source_feature, target_feature)
                    loss_dis = self.loss_coral_fun(source_feature, target_feature)
                    # loss_dis = self.loss_moment_func(source_feature, target_feature, 5)
                    loss = loss_cls + self.tranfer_lambda * loss_dis
                    # loss = loss_cls
                else:
                    loss = loss_cls

                # img_test1 = img_s.cpu()[0, :, :, :].squeeze().permute(1, 2, 0).numpy()
                # img_test2 = img_t.cpu()[0, :, :, :].squeeze().permute(1, 2, 0).numpy()
                # plt.subplot(121)
                # plt.imshow(img_test1)
                # plt.subplot(122)
                # plt.imshow(img_test2)
                # plt.show()
                # if (np.any(np.isnan(img_test))):
                # plt.imshow(img_test)
                # plt.show()
                #     print("batch_index:", batch_idx, "probs:", log_probs,\
                #           'loss_cls:', loss_cls, 'loss_dis:', loss_dis, 'loss_total:', loss)
                loss.backward()
                optimizer.step()
                if self.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tCls Loss: {:.6f} Dis Loss: {:.6f} Total Loss: {:.6f}'.format(
                        iter, batch_idx * len(img_s), len(self.dataset_loader_train),
                               100. * batch_idx * len(img_s) / len(self.dataset_loader_train),
                               loss_cls.item(), loss_dis.item(), loss.item()))
                batch_loss['cls'].append(loss_cls.item())
                batch_loss['dis'].append(loss_dis.item())
                batch_loss['total'].append(loss.item())
            epoch_loss['cls'].append(sum(batch_loss['cls'])/len(batch_loss['cls']))
            epoch_loss['dis'].append(sum(batch_loss['dis']) / len(batch_loss['dis']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
        return_loss = {'cls':sum(epoch_loss['cls']) / len(epoch_loss['cls']),
                       'dis':sum(epoch_loss['dis']) / len(epoch_loss['dis']),
                       'total':sum(epoch_loss['total']) / len(epoch_loss['total'])}
        return net.state_dict(), return_loss

    def train_mul(self, net, w_locals, source_domain, source_client):
        net.train()
        net_copy = copy.deepcopy(net).to(self.device)
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum)

        epoch_loss = {'cls':[], 'dis':[], 'total':[]}
        for iter in range(self.local_ep):
            batch_loss = {'cls':[], 'dis':[], 'total':[]}
            for batch_idx, data in enumerate(self.dataset_loader_train):
                # img_t = autograd.Variable(data['target_imgs'].to(self.device)) #列表
                # img_t = data['target_imgs']
                img_s = autograd.Variable(data[source_domain + '_imgs'][source_client].to(self.device))
                label_s = autograd.Variable(data[source_domain + '_labels'][source_client].long().to(self.device))
                # images, labels = images.to(self.args.device), labels.to(self.args.device)

                net_copy.load_state_dict(w_locals[source_domain][source_client])
                target_feature = net_copy(data[source_domain + '_imgs'][source_client].to(self.device))[0]
                target_num = 1
                for domain_idx in w_locals:
                    for client_idx in range(len(w_locals[domain_idx])):
                        if domain_idx != source_domain or client_idx != source_client:
                            net_copy.load_state_dict(w_locals[domain_idx][client_idx])
                            target_feature += net_copy(data[domain_idx + '_imgs'][client_idx].to(self.device))[0]
                            target_num += 1

                target_feature /= torch.tensor(target_num, dtype=torch.float32)
                target_feature = target_feature.detach()

                # net.load_state_dict(w_locals[source_domain][source_client])
                # net.train()
                net.zero_grad()
                source_feature, log_probs = net(img_s)

                # target_feature /= torch.tensor(len(img_t) + 1, dtype=torch.float32)

                loss_cls = self.loss_cls_func(log_probs, label_s)

                # loss_dis = self.euclidean(source_feature, target_feature)
                loss_dis = torch.zeros(1)
                # self.loss_dis_func(source_feature, target_feature)
                # loss_dis = self.loss_cls_func(log_probs, label_s)
                if self.transfer:
                    # loss_dis = self.loss_mmd_func(source_feature, target_feature)
                    loss_dis = self.loss_coral_fun(source_feature, target_feature)
                    # loss_dis = self.loss_moment_func(source_feature, target_feature, 5)
                    loss = loss_cls + self.tranfer_lambda * loss_dis
                    # loss = loss_cls
                else:
                    loss = loss_cls

                # img_test1 = img_s.cpu()[0, :, :, :].squeeze().permute(1, 2, 0).numpy()
                # img_test2 = img_t.cpu()[0, :, :, :].squeeze().permute(1, 2, 0).numpy()
                # plt.subplot(121)
                # plt.imshow(img_test1)
                # plt.subplot(122)
                # plt.imshow(img_test2)
                # plt.show()
                # if (np.any(np.isnan(img_test))):
                # plt.imshow(img_test)
                # plt.show()
                #     print("batch_index:", batch_idx, "probs:", log_probs,\
                #           'loss_cls:', loss_cls, 'loss_dis:', loss_dis, 'loss_total:', loss)
                loss.backward()
                optimizer.step()
                if self.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tCls Loss: {:.6f} Dis Loss: {:.6f} Total Loss: {:.6f}'.format(
                        iter, batch_idx * len(img_s), len(self.dataset_loader_train),
                               100. * batch_idx * len(img_s) / len(self.dataset_loader_train),
                               loss_cls.item(), loss_dis.item(), loss.item()))
                batch_loss['cls'].append(loss_cls.item())
                batch_loss['dis'].append(loss_dis.item())
                batch_loss['total'].append(loss.item())
            epoch_loss['cls'].append(sum(batch_loss['cls'])/len(batch_loss['cls']))
            epoch_loss['dis'].append(sum(batch_loss['dis']) / len(batch_loss['dis']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
        return_loss = {'cls':sum(epoch_loss['cls']) / len(epoch_loss['cls']),
                       'dis':sum(epoch_loss['dis']) / len(epoch_loss['dis']),
                       'total':sum(epoch_loss['total']) / len(epoch_loss['total'])}
        return net.state_dict(), return_loss

    def euclidean(self, f1, f2):
        # f1_temp = f1.cpu().numpy()
        # f2_temp = f2.cpu().numpy()
        # print(f1, f2)

        return torch.sqrt(torch.sum(torch.pow(f1.mean(0) - f2.mean(0), 2)))
        # return ((f1 - f2)**2).sum().sqrt() + 1e-6

    def feature_mean(self, net, data):
        pass