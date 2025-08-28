import gc
import logging
from typing import Dict

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class FlyPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(FlyPrompt, self).__init__(*args, **kwargs)

        self.task_id = 0
        # self.label_to_task = {}
        self.label_to_task: Dict[int, set] = {}

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        self.update_statistics(images.clone(), labels.clone())

        del images, labels
        gc.collect()
        return _loss / _iter, _acc / _iter
    
    def update_statistics(self, images, labels):
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())

        unique_labels = torch.unique(labels)
        for label in unique_labels:
            if label.item() not in self.label_to_task:
                # self.label_to_task[label.item()] = self.task_id
                self.label_to_task[label.item()] = set()
            self.label_to_task[label.item()].add(self.task_id)

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)

        self.optimizer.zero_grad()
        if not self.no_batchmask:
            logit, loss = self.model_forward(x,y,mask=logit_mask)
        else:
            logit, loss = self.model_forward(x,y)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                # # use label_to_task to get cheat expert_ids
                # expert_ids = [self.label_to_task[label.item()] for label in y]
                # expert_ids = torch.tensor(expert_ids, device=y.device, dtype=torch.long)

                # logit = self.model(x, expert_ids=expert_ids)

                # use label_to_task to get cheat expert_ids - dual expert ensemble
                expert_ids_1 = []
                expert_ids_2 = []

                for y_i in y:
                    task_set = self.label_to_task.get(y_i.item(), {self.task_id})
                    task_list = sorted(list(task_set))

                    expert_ids_1.append(task_list[0])

                    if len(task_list) > 1:
                        expert_ids_2.append(task_list[1])
                    else:
                        expert_ids_2.append(task_list[0])  # Duplicate first seen task

                expert_ids_1 = torch.tensor(expert_ids_1, device=y.device, dtype=torch.long)
                expert_ids_2 = torch.tensor(expert_ids_2, device=y.device, dtype=torch.long)
                logit_1 = self.model(x, expert_ids=expert_ids_1)
                logit_2 = self.model(x, expert_ids=expert_ids_2)

                logit = (logit_1 + logit_2) / 2.0

                logit = logit + self.mask
                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        self.model_without_ddp.process_task_count()
        self.task_id += 1