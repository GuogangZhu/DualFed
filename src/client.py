import copy
import torch
from utils.Loss import Contrastive_Loss

class Client:
    def __init__(self, local_model, args):
        # local model
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.local_model = local_model
        self.local_model.to(self.device)

        self.args = args

        # define optimizer
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.local_model.parameters, lr=self.args.lr, weight_decay=1e-4)
        else:
            raise NotImplementedError

        # define loss function
        self.loss_cls_func = torch.nn.CrossEntropyLoss()
        self.loss_con_func = Contrastive_Loss(temperature = self.args.con_temp)


    # freeze or unfreeze specific layers in layer_list
    def opt_layers(self, layer_list, mode='frozen'):
        layers = layer_list.split(',')

        if mode == 'frozen':
            frozen_flag = True
        else:
            frozen_flag = False

        for name, param in self.local_model.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = not frozen_flag
            else:
                param.requires_grad = frozen_flag

        trained_param = list(filter(lambda p: p.requires_grad is True, self.local_model.parameters()))

        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(trained_param, lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(trained_param, lr=self.args.lr, weight_decay=1e-4)
        else:
            raise NotImplementedError

    # iteratively performing local training
    def train(self, train_dataloader):
        epoch_loss = {'local_cls': [], 'global_cls':[], 'con': []}

        # freeze global classifier and update main branch
        self.opt_layers('global_C', mode='frozen')

        for iter in range(self.args.local_ep):
            self.local_model.train()

            batch_loss = {'local_cls': [], 'con': []}
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(self.device), labels.long().to(self.device)

                self.optimizer.zero_grad()
                local_features, global_features, local_probs, global_probs = self.local_model(images)

                loss_cls_local = self.loss_cls_func(local_probs, labels)

                loss_con = self.args.con_lambda * self.loss_con_func(local_features, labels)

                loss = loss_cls_local +  loss_con

                loss.backward()
                self.optimizer.step()

                batch_loss['local_cls'].append(loss_cls_local.item())
                batch_loss['con'].append(loss_con.item())

            for k in batch_loss:
                epoch_loss[k].append(sum(batch_loss[k]) / len(batch_loss[k]))

        # freeze main branch and update global classifier
        self.opt_layers('global_C', mode='train')
        for iter in range(self.args.local_ep):
            batch_loss = {'global_cls': []}
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(self.device), labels.long().to(self.device)

                self.optimizer.zero_grad()
                local_features, global_features, local_probs, global_probs = self.local_model(images)

                loss_cls_global = self.loss_cls_func(global_probs, labels)

                loss = loss_cls_global

                loss.backward()
                self.optimizer.step()

                batch_loss['global_cls'].append(loss_cls_global.item())

            for k in batch_loss:
                epoch_loss[k].append(sum(batch_loss[k]) / len(batch_loss[k]))

        return {k: sum(epoch_loss[k]) / len(epoch_loss[k]) for k in epoch_loss}

    def inference(self, dataloader, model=None):
        if model is None:
            model = copy.deepcopy(self.local_model)

        model.eval()

        local_loss, global_loss, total, local_correct, global_correct, ens_correct = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.long().to(self.device)

            # inference
            _, _, local_probs, global_probs = model(images)

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



