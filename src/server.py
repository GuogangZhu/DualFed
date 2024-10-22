import sys
sys.path.append("..")
import copy
import os
import torch
import dataset
from src.client import Client
from models.Nets import BuildModel
from utils.save_info import save_para
from utils.file_operation import mkdir
import time
import json
import logging
import random
from utils.metric_logger import MetricLogger

class Server:
    def __init__(self, args, round):
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

        self.domains = args.domains.split(',')

        self.args = args

        # initialize global model
        self.global_model = BuildModel(model_name=args.model, num_classes=args.num_classes, device=self.device)

        self.global_weights = self.global_model.state_dict()

        self.round = round

        self.total_clients = len(self.domains) * self.args.num_users

        # generate global dataset
        self.get_global_dataset(self.domains)

        # generate clients [domain: [client 1, client 2, ... , client N]]
        self.clients = {domain: [Client(copy.deepcopy(self.global_model), args) for i in range(args.num_users)] for domain in self.domains}

        # inference loss
        self.loss_cls_func = torch.nn.CrossEntropyLoss()

        # create record file direction
        if (self.args.save_dir != ''):
            mkdir(self.args.save_dir)
        else:
            now_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
            self.args.save_dir = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}/save/{now_time}_{args.domains}_' \
                    f'{args.local_layers}_{args.local_bs}_{args.local_ep}_{args.train_num}_{args.num_users}_' \
                    f'_cls{args.cls_scale}_dis{args.dis_scale}_div{args.div_scale}_{random.randint(100, 999)}/'
            mkdir(self.args.save_dir)

        # create model direction
        mkdir(f'{self.args.save_dir}model/')

        # create log direction
        mkdir(f'{self.args.save_dir}log/')

        self.metric_logger = MetricLogger(args)

        # create logger
        self.logger = logging.getLogger()
        self.fh = None
        self.ch = None

    def get_global_dataset(self, domains):
        # get training and test dataset
        # data structure: {domain1: dataset1, domain2: dataset2, ...}
        dataset_read = getattr(dataset, self.args.dataset)
        self.dataset_train_source, self.dataset_test_source = dataset_read(
            domains=domains, base_dir=self.args.data_dir, train_num=self.args.train_num,
            test_num=self.args.test_num, scale=self.args.scale
        )

        # generate train dataset loader for each client in each domain
        # data structure: {domain: [client 1, client 2, client 3, ...]}
        self.dataset_train_loader, self.dataset_train_len = dataset.dataset_loader(
            datasets=self.dataset_train_source, bs=self.args.local_bs,
            num_users=self.args.num_users, iid_sampling=True
        )

        # clients in the same domain share an identical dataset
        self.dataset_test_loader, self.dataset_test_len = dataset.dataset_loader(
            datasets=self.dataset_test_source, bs=self.args.local_bs,
            num_users=1, iid_sampling=True
        )

    def average_weights(self):
        self.global_weights = copy.deepcopy(self.clients[self.domains[0]][0].local_model.state_dict())
        for key in self.global_weights.keys():
            for domain in self.domains:
                for client in range(self.args.num_users):
                    if domain == self.domains[0] and client == 0: continue
                    self.global_weights[key] += self.clients[domain][client].local_model.state_dict()[key]
            self.global_weights[key] = torch.div(self.global_weights[key], self.total_clients)

    def send_parameters(self):
        if self.args.local_layers == '':  # collaborate train a global model
            for domain in self.domains:
                for client in range(self.args.num_users):
                    self.clients[domain][client].local_model.load_state_dict(self.global_weights)
        elif self.args.local_layers == 'all':   # local training
            return
        else:
            local_layers = self.args.local_layers.split(',') # personalizing layers that contains substring in local_layers
            for domain in self.domains:
                for client in range(self.args.num_users):
                    temp_state = copy.deepcopy(self.clients[domain][client].local_model.state_dict())
                    for key in self.global_weights.keys():
                        if not any(layer in key for layer in local_layers):
                            temp_state[key] = self.global_weights[key]
                    self.clients[domain][client].local_model.load_state_dict(temp_state)

    def train(self):
        save_para(self.args, f'{self.args.save_dir}training_config.txt')

        self.current_epoch = 0
        for epoch in range(self.args.epochs):
            self.current_epoch += 1

            print(f'Start Training round: {epoch}')

            # local training on each client
            for domain in self.domains:
                for client in range(self.args.num_users):
                    loss = self.clients[domain][client].train(train_dataloader=self.dataset_train_loader[domain][client])
                    self.metric_logger.update(self.current_epoch, **{f'train_{k}_{domain}_client{client}': v for k, v in loss.items()})

            # aggregate local weight
            self.average_weights()

            # broadcast aggregated weight
            self.send_parameters()

            # compute training metrics:
            if self.args.test_step > 0 and (epoch + 1) % self.args.test_step == 0:
                self.training_metric()

            # get training information
            if self.args.log_step > 0 and (epoch + 1) % self.args.log_step == 0:
                metric_list = ['train_local_cls', 'train_global_cls', 'train_dis', 'test_acc']
                self.metric_logger.get_log(epoch, 'stage', metric_list=metric_list)

            if self.args.model_step > 0 and (epoch + 1) % self.args.model_step == 0:
                self.save_model(epoch)

        # get final log
        metric_list = ['best_acc']
        self.metric_logger.get_log(self.args.epochs, 'final', metric_list=metric_list)

        # save training log
        self.save_res()

        # print all log
        self.metric_logger.print_log()
        self.metric_logger.update_final_best()

        # remove handle for logging
        self.metric_logger.close()

        return

    def inference(self, dataloader):
        self.global_model.eval()
        local_loss, global_loss, total, local_correct, global_correct, ens_correct = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.long().to(self.device)

            # inference
            _, _, local_probs, global_probs = self.global_model(images)

            loss_cls_local = self.loss_cls_func(local_probs, labels)
            loss_cls_global = self.loss_cls_func(global_probs, labels)

            local_loss += loss_cls_local.item()
            global_loss += loss_cls_global.item()

            # prediction
            _, local_pred = torch.max(local_probs, 1)
            _, global_pred = torch.max(global_probs, 1)
            _, ens_pred = torch.max(local_probs + global_probs, 1)
            local_pred = local_pred.view(-1)
            global_pred = global_pred.view(-1)
            ens_pred = ens_pred.view(-1)

            # accuracy
            global_correct += torch.sum(torch.eq(global_pred, labels.long())).item()
            local_correct += torch.sum(torch.eq(local_pred, labels.long())).item()
            ens_correct += torch.sum(torch.eq(ens_pred, labels.long())).item()
            total += len(labels)

        global_accuracy = global_correct / total
        local_accuracy = local_correct / total
        accuracy = ens_correct / total

        local_loss = local_loss / total
        global_loss = global_loss / total
        return {'local_acc': local_accuracy, 'global_acc': global_accuracy, 'acc': accuracy,
                'local_cls': local_loss, 'global_cls': global_loss}

    def training_metric(self):
        # test on each client
        for domain in self.domains:
            for client in range(self.args.num_users):
                # # train dataset
                # res = self.clients[domain][client].inference(dataloader=self.dataset_train_loader[domain][client])
                # self.metric_logger.update(self.current_epoch, **{f'train_{k}_{domain}_client{client}': v for k, v in res.items()})

                # test dataset
                res = self.clients[domain][client].inference(dataloader=self.dataset_test_loader[domain][0])
                self.metric_logger.update(self.current_epoch, **{f'test_{k}_{domain}_client{client}': v for k, v in res.items()})

                # update best accuracy
                for k in res.keys():
                    if 'acc' in k:
                        self.update_best_acc(key=f'{k}_{domain}_client{client}', domain=domain, client=client)

    def update_best_acc(self, key, domain, client):
        # update best accuracy
        save_flag = self.metric_logger.update_best(key)
        # update best model
        if self.args.best_model == True and save_flag == True:
            torch.save(self.clients[domain][client].local_model.state_dict(),
                       f'{self.args.save_dir}model/{key}_best_round{self.round}.pb')

    def save_res(self):
        log_dict = self.metric_logger.meters

        log_json = json.dumps(log_dict)

        fp = open(f'{self.args.save_dir}log/Training_metric_{self.args.dataset}_round_{self.round}.json', 'w')
        fp.write(log_json)
        fp.close()

    def save_model(self, epoch):
        # save client local model
        for domain in self.domains:
            for client in range(self.args.num_users):
                torch.save(self.clients[domain][client].local_model.state_dict(),
                           f'{self.args.save_dir}model/{domain}_client{client}_epoch{epoch}_round{self.round}.pb')
        # save global model
        torch.save(self.global_weights, f'{self.args.save_dir}model/global_epoch{epoch}_round{self.round}.pb')
        return