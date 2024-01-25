import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter

class WeightedLoss(nn.Module):
    def __init__(self, task_num):
        super(WeightedLoss, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, loss_1, loss_2):
        precision_1 = torch.exp(-self.log_vars[0])
        loss_1 = precision_1 * loss_1 + self.log_vars[0]

        precision_2 = torch.exp(-self.log_vars[1])
        loss_2 = precision_2 * loss_2 + self.log_vars[1]

        return loss_1 + loss_2


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        #self.tasks = nn.ModuleList(tasks)
        self.sigma_1 = nn.Parameter(torch.zeros(1))
        self.sigma_2 = nn.Parameter(torch.zeros(1))
        #self.mse = nn.MSELoss()

    def forward(self, loss_1, loss_2):
       #l = [self.mse(f(x), y) for y, f in zip(targets, self.tasks)]

       l1 = (torch.Tensor(loss_1) / self.sigma_1**2) + self.sigma_1
       l2 = (torch.Tensor(loss_2) / self.sigma_2 ** 2) + self.sigma_2
       loss = l1 + l2
       return loss

#multitask_loss = MultiTaskLoss()
multitask_loss = WeightedLoss(task_num=2)

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss1_meter = AverageValueMeter()
        loss2_meter = AverageValueMeter()
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        #metrics_meters2 = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, z in iterator:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                loss1, loss2, weighted_loss, y_pred, z_pred = self.batch_update(x, y, z)

                # update loss logs
                loss1_value = loss1.cpu().detach().numpy()
                loss2_value = loss2.cpu().detach().numpy()
                loss_value = weighted_loss.cpu().detach().numpy()

                loss1_meter.add(loss1_value)
                loss2_meter.add(loss2_value)
                loss_meter.add(loss_value)

                loss1_logs = {self.loss.__name__ + str('_breast'):loss1_meter.mean}
                loss2_logs = {self.loss.__name__ + str('_dense'):loss2_meter.mean}
                loss_logs = {self.loss.__name__ + str('_weighted'): loss_meter.mean}

                logs.update(loss1_logs)
                logs.update(loss2_logs)
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value1 = metric_fn(y_pred, y).cpu().detach().numpy()
                    metric_value2 = metric_fn(z_pred, z).cpu().detach().numpy()

                    metric_value = (metric_value1 + metric_value2)/2

                    metrics_meters[metric_fn.__name__ ].add(metric_value)
                    #metrics_meters2[metric_fn.__name__ ].add(metric_value2)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                #metrics_logs2 = {k: v.mean for k, v in metrics_meters2.items()}
                #logs.update(metrics_logs1)
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, lr_schedular, device='cuda', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.lr_schedular = lr_schedular

        #self.lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, z):
        self.optimizer.zero_grad()
        prediction1, prediction2 = self.model.forward(x)
        loss1 = self.loss(prediction1, y)
        loss2 = self.loss(prediction2, z)

        #weighted_loss = multitask_loss(loss1, loss2)
        #weighted_loss = weighted_loss.to('cuda')
        weighted_loss = 0.5*loss1 + 0.5*loss2

        #loss = 0.5*loss1 + 0.5*loss2
        #loss = awl(loss1, loss2)

        weighted_loss.backward()
        self.optimizer.step()
        #self.lr_schedular.step()
        return loss1, loss2, weighted_loss, prediction1, prediction2


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cuda', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, z):
        with torch.no_grad():
            prediction1, prediction2 = self.model.forward(x)
            loss1 = self.loss(prediction1, y)
            loss2 = self.loss(prediction2, z)

            #weighted_loss = multitask_loss(loss1, loss2)
            #weighted_loss = weighted_loss.to('cuda')
            weighted_loss = 0.5*loss1 + 0.5*loss2

            #loss = 0.5*loss1 + 0.5*loss2
            
        return loss1, loss2, weighted_loss, prediction1, prediction2
