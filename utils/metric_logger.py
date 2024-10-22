# -*- coding: utf-8 -*-
# @Time    : 2023/11/28 17:28
# @Author  : Guogang Zhu
# @File    : metric_logger.py
# @Software: PyCharm

import logging
import torch
import wandb
from typing import Iterable, Dict, List

class MetricLogger(object):

    def __init__(self, args):
        self.args = args

        # recording metrics
        self.meters = {}

        # creating logger
        self.init_log()

    def init_log(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.NOTSET)

        self.log_file = f'{self.args.save_dir}log/Training_log_{self.args.dataset}_round_{self.args.current_round}.log'

        self.fh = logging.FileHandler(self.log_file, mode='a')
        self.fh.setLevel(logging.NOTSET)

        formatter = logging.Formatter("%(message)s")
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)

        if self.args.verbose == 1:
            self.ch = logging.StreamHandler()
            self.ch.setLevel(logging.NOTSET)
            self.ch.setFormatter(formatter)
            self.logger.addHandler(self.ch)

        self.logger.info(f'Communication Rounds: {self.args.epochs}')
        self.logger.info(f'Client Number for Each Domain: {self.args.num_users}')
        self.logger.info(f'Local Epochs: {self.args.local_ep}')
        self.logger.info(f'Local Batch Size: {self.args.local_bs}')
        self.logger.info(f'Learning Rate: {self.args.lr}')
        self.logger.info(f'Model Type: {self.args.model}')
        self.logger.info(f'Domains: {self.args.domains}')
        self.logger.info(f'Local Layers: {self.args.local_layers}')

    def update(self, current_epoch, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters.keys():
                self.meters[k] = []
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].append(v)
            wandb.log({k: v}, step=current_epoch)

    def update_best(self, key):
        best_name = f'best_{key}'
        test_name = f'test_{key}'
        if best_name not in self.meters.keys():
            if test_name in self.meters.keys():
                self.meters[best_name] = max(self.meters[test_name])
            else:
                self.meters[best_name] = 0.0
            return True
        else:
            if self.meters[best_name] < self.meters[test_name][-1]:
                self.meters[best_name] = self.meters[test_name][-1]
                return True
            return False

    def update_final_best(self):
        for k in self.meters.keys():
            if 'best_acc' in k:
                wandb.summary[f'final_{k}'] = self.meters[k]

    def close(self):
        if self.fh is not None:
            self.logger.removeHandler(self.fh)

        if self.ch is not None:
            self.logger.removeHandler(self.ch)

    def get_log(self, epoch=0, mode='stage', metric_list=['best_acc']):
        self.domains = self.args.domains.split(',')
        #last round
        if mode == 'final':
            # metric list on each client
            log_str = self.get_log_body(metric_list, global_flag=False)
            log_str.insert(0, self.get_log_head('Final Accuracy for Each Client of Each Domain', len(log_str[0])))
            log_str.append(self.get_log_tail(len(log_str[-1])))

            for line in log_str:
                self.logger.info(line)

            log_str = self.get_log_body(metric_list, global_flag=True)
            log_str.insert(0, self.get_log_head('Final Global Accuracy for Each Domain', len(log_str[0])))
            log_str.append(self.get_log_tail(len(log_str[-1])))
            for line in log_str:
                self.logger.info(line)

        #training stage
        elif mode == 'stage':
            log_str = self.get_log_body(metric_list, global_flag=False)
            log_str.insert(0, self.get_log_head('Communication Round: {:3}'.format(epoch), len(log_str[0])))
            log_str.append(self.get_log_tail(len(log_str[-1])))

            for line in log_str:
                self.logger.info(line)

    def get_log_body(self, metric_list, global_flag=False):
        log_str = []
        if global_flag == True:
            for domain in self.domains:
                temp_str = '|Domain: {:10} |'.format(domain)
                for metric in metric_list:
                    if f'{metric}_{domain}_global' in self.meters.keys():
                        if isinstance(self.meters[f'{metric}_{domain}_global'], List):
                            metric_value = self.meters[f'{metric}_{domain}_global'][-1]
                        else:
                            metric_value = self.meters[f'{metric}_{domain}_global']
                        temp_str += ' {}:{:.6f} |'.format(metric, metric_value)
                log_str.append(temp_str)
        else:
            for domain in self.domains:
                for client in range(self.args.num_users):
                    temp_str = '|Domain: {:10} | Client: {:2} |'.format(domain, client)
                    for metric in metric_list:
                        if f'{metric}_{domain}_client{client}' in self.meters.keys():
                            if isinstance(self.meters[f'{metric}_{domain}_client{client}'], List):
                                metric_value = self.meters[f'{metric}_{domain}_client{client}'][-1]
                            else:
                                metric_value = self.meters[f'{metric}_{domain}_client{client}']
                            temp_str += ' {}:{:.6f} |'.format(metric, metric_value)
                    log_str.append(temp_str)
        return log_str

    def get_log_head(self, str, total_len):
        _len1 = (total_len - len(str) - 2) // 2
        _len2 = total_len - len(str) - 2 - _len1
        return '|' + '-' * (_len1) + str + '-' * (_len2) + '|'

    def get_log_tail(self, total_len):
        return '|' + '-' * (total_len - 2) + '|'

    def print_log(self):
        fp = open(self.log_file)
        lines = fp.read().splitlines()
        for line in lines:
            print(line)
        fp.close()

