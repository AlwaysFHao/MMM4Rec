# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2025/1/17

import os.path
import time

import lightning as L
import torch
import gc
from tabulate import tabulate
from tqdm import tqdm
from lightning_fabric.loggers import CSVLogger, TensorBoardLogger

from callback import EarlyStopping, save_class_code
try:
    from .utils import calc_recall, calc_ndcg, set_color, get_gpu_usage, get_logger, get_config_str, get_current_time_info
except:
    from utils import calc_recall, calc_ndcg, set_color, get_gpu_usage, get_logger, get_config_str, get_current_time_info

class PreTrainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 config,
                 checkpoint=None,
                 accelerator="auto",
                 strategy="auto",
                 devices=1,
                 precision="32-true",
                 save_model_classes=None):

        self.model = model
        self.model_name = model.__class__.__name__
        self.save_model_classes = save_model_classes
        assert isinstance(save_model_classes, list) or save_model_classes is None, '"save_model_classes" can only be a list or none! '

        self.dataset_name = config['dataset_path'].split('/')[-1]

        self.optimizer = optimizer
        if 'is_scheduler' in config.keys() and config['is_scheduler'] is True:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            self.is_scheduler = True
        else:
            self.is_scheduler = False

        self.max_epochs = config['Epoch']
        if 'stop_epoch' in config.keys():
            self.early_stop = EarlyStopping(patience=config['stop_epoch'])
        else:
            self.early_stop=None
        self.valid_metric = config['valid_metric']
        if 'metrics_table_displays' in config.keys():
            self.metrics_table_displays = config['metrics_table_displays']
        else:
            self.metrics_table_displays = True

        if 'is_pbar' in config.keys():
            self.is_pbar = config['is_pbar']
        else:
            self.is_pbar = True

        if 'save_step' in config.keys():
            self.save_step = int(config['save_step'])
        else:
            self.save_step = 10

        if 'light_save' in config.keys():
            self.light_save = config['light_save']
        else:
            self.light_save = False

        self.metrix = config['metrics']
        for m in self.metrix:
            assert m in ['Hit', 'NDCG'], f"Invalid value {m}, the value of metrix must be within ['Hit', 'NDCG']!"
        self.top_k = config['top_k']
        assert all(map(lambda x: isinstance(x, int), self.top_k)), "Not all elements in top_k are integers!"

        self.best_val_score = None
        self._create_root_save_folder()
        self.save_folder = self._create_save_folder()
        self.save_path = self._create_save_path()
        self.model_save_path = self._create_model_save_path()

        self.start_epoch = 0
        self.best_epoch = 0

        csv_logger = CSVLogger(root_dir=self.save_path, flush_logs_every_n_steps=1)
        tensorboard_logger = TensorBoardLogger(root_dir=self.save_path)
        self.fabric = L.Fabric(accelerator=accelerator,
                               strategy=strategy,
                               devices=devices,
                               precision=precision,
                               loggers=[csv_logger, tensorboard_logger])
        self.fabric.launch()

        if self.fabric.global_rank == 0:
            self.logger = get_logger(f'{self.model_name}', os.path.join(self.save_path, 'output.log'))
            self.logger.info(f'{get_config_str(config)}')
            self.logger.info(set_color(f'{model}', color='white'))
            self.save_model_code()

        if 'pretrain_checkpoint' in config.keys():
            self._load_pretrain_checkpoint(config['pretrain_checkpoint'])

        if 'model_compile' in config.keys() and config['model_compile'] is True:
            self.model.pre_compile()

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        if checkpoint is not None:
            self._resume_checkpoint(checkpoint)

        if 'accumulation_steps' in config.keys():
            self.accumulation_steps = config['accumulation_steps']
        else:
            self.accumulation_steps = 1

        if 'return_scores' in config.keys():
            self.return_scores = config['return_scores']
        else:
            self.return_scores = False

    def save_model_code(self):
        save_class_code(self.model.__class__, self.model_save_path, self.logger)
        if self.save_model_classes is not None:
            for cls in self.save_model_classes:
                save_class_code(cls, self.model_save_path, self.logger)

    @staticmethod
    def _create_root_save_folder():
        root_save_folder = os.path.join(os.getcwd(), 'saved')
        if not os.path.exists(root_save_folder):
            os.mkdir(root_save_folder)

    def _create_save_folder(self):
        save_folder = os.path.join(os.getcwd(), 'saved', self.model_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_folder = os.path.join(save_folder, self.dataset_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        return save_folder

    def _create_save_path(self):
        save_path = os.path.join(self.save_folder, get_current_time_info())
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        return save_path

    def _create_model_save_path(self):
        model_save_path = os.path.join(self.save_path, 'model_code')
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        return model_save_path

    def _resume_checkpoint(self, checkpoint):
        full_checkpoint = self.fabric.load(checkpoint)
        self.model.load_state_dict(full_checkpoint["model"], strict=False)
        self.optimizer.load_state_dict(full_checkpoint["optimizer"])
        self.start_epoch = full_checkpoint["epoch"]
        if self.is_scheduler:
            self.scheduler.load_state_dict(full_checkpoint["scheduler"])  # 恢复调度器

        if 'best_val_score' in full_checkpoint.keys():
            self.best_val_score = full_checkpoint["best_val_score"]
        else:
            pass
        if self.early_stop is not None:
            self.early_stop.best_score = self.best_val_score
        if self.fabric.global_rank == 0:
            self.logger.info(set_color(f'load weights from: {checkpoint}, resume epoch: {self.start_epoch}, best val score: {self.best_val_score}', color='white'))

    def _load_pretrain_checkpoint(self, checkpoint):
        model_checkpoint = self.fabric.load(checkpoint)["model"]
        model_checkpoint.pop('text_item_embedding_bias', None)
        model_checkpoint.pop('vision_item_embedding_bias', None)
        model_checkpoint.pop('text_embedding', None)
        model_checkpoint.pop('vision_embedding', None)
        self.model.load_state_dict(model_checkpoint, strict=False)
        if self.fabric.global_rank == 0:
            self.logger.info(set_color(f'load pretrained weights from: {checkpoint}', color='white'))


    def get_metrics(self, sort_lists, batch_count):
        matrix_dict = {}
        for m in self.metrix:
            for k in self.top_k:
                if m == 'Hit':
                    matrix_dict[f'Hit@{k}'] = calc_recall(sort_lists, batch_count, k)
                elif m == 'NDCG':
                    matrix_dict[f'NDCG@{k}'] = calc_ndcg(sort_lists, batch_count, k)
                else:
                    raise ValueError(f'Metric {m} is not supported temporarily!')
        return matrix_dict


    def fit(self, train_dataloader):

        model = self.model
        optimizer = self.optimizer

        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)

        for epoch in range(self.max_epochs):
            torch.cuda.empty_cache()
            if self.fabric.global_rank == 0:
                self.logger.info('\n-----------------***************** new epoch *****************-----------------\n', )
            model.train()
            training_step_outputs = []
            validation_step_outputs = []
            batch_count = 0

            if self.fabric.global_rank == 0 and self.is_pbar:
                pbar = tqdm(total=len(train_dataloader),
                            desc=set_color(f'Training Epoch: {epoch + 1 + self.start_epoch}', 'blue'),
                            colour='white')
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.fabric.device)
                self.fabric.barrier()
            ts = time.perf_counter()

            optimizer.zero_grad()

            for i, batch in enumerate(train_dataloader):
                # 梯度累计
                is_accumulating = (i + 1) % self.accumulation_steps == 0 or i == len(train_dataloader) - 1
                # 多设备高效梯度累计
                with self.fabric.no_backward_sync(model, enabled=is_accumulating):
                    if self.return_scores:
                        loss, scores_index, batch_num = model.training_step(batch, return_scores=self.return_scores)
                    else:
                        loss = model.training_step(batch, return_scores=self.return_scores)

                    self.fabric.backward(loss)
                if is_accumulating:
                    optimizer.step()
                    optimizer.zero_grad()
                training_step_outputs.append(loss.detach())
                if self.return_scores:
                    validation_step_outputs.append(scores_index.detach())
                    batch_count += batch_num
                del batch

                if self.fabric.global_rank == 0 and self.is_pbar:
                    pbar.update(1)

            if self.is_scheduler:
                # 学习率衰减
                self.scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize(self.fabric.device)
                self.fabric.barrier()
            td = time.perf_counter()
            with torch.no_grad():
                loss_pre = torch.stack(training_step_outputs)
                del training_step_outputs
                # loss_pre = self.fabric.all_reduce(loss_pre, reduce_op="mean")
                loss_sum = loss_pre.sum()
                loss_mean = loss_pre.mean()
                if self.return_scores:
                    sort_lists = torch.cat(validation_step_outputs, dim=0)
                    del validation_step_outputs
                    metrics = self.get_metrics(sort_lists, batch_count)
                    # metrics = self.fabric.all_reduce(metrics, reduce_op='mean')
                    val_score = metrics[self.valid_metric]

            if self.fabric.global_rank == 0:
                if torch.cuda.is_available():
                    gpu_usage = set_color("GPU RAM: " + get_gpu_usage(self.fabric.device), "yellow")
                    if self.is_pbar:
                        pbar.set_postfix_str(gpu_usage)
                        pbar.close()
                    self.logger.info(set_color(f'Training Epoch {epoch + 1 + self.start_epoch}, Training Time {(td - ts):.5f}s, \nTraining Total Loss {loss_sum}, Training Mean Loss {loss_mean}, {gpu_usage}', color='white'))
                else:
                    if self.is_pbar:
                        pbar.close()
                    self.logger.info(set_color(f'Training Epoch {epoch + 1 + self.start_epoch}, Training Time {(td - ts):.5f}s, \nTraining Total Loss {loss_sum}, Training Mean Loss {loss_mean}', color='white'))
                # 分数矩阵
                if self.return_scores:
                    if self.metrics_table_displays:
                        self.logger.info(set_color(
                            f'\n{tabulate([list(metrics.values())], headers=list(metrics.keys()), tablefmt="fancy_grid")}',
                            color='green'))
                    else:
                        self.logger.info(set_color(
                            f'\nMetrics: ' + ', '.join(
                                f'{k}={metrics[k]} ' for k in metrics.keys()),
                            color='green'))


            if self.fabric.global_rank == 0:
                self.best_epoch = epoch + 1 + self.start_epoch
                state = {"model": model,
                         "optimizer": optimizer,
                         "epoch": epoch + 1 + self.start_epoch}
                if self.is_scheduler:
                    state['scheduler'] = self.scheduler
                save_path = os.path.join(self.save_path, 'last_epoch_model.ckpt')

                filter = {"model": lambda k, v: "text_embedding" not in k and "vision_embedding" not in k}

                self.fabric.save(save_path, state, filter=filter)

                if (epoch + 1 + self.start_epoch) % self.save_step == 0:
                    save_path = os.path.join(self.save_path, f'the_{epoch + 1 + self.start_epoch}_epoch_model.ckpt')
                    if self.light_save:
                        filter = {"model": lambda k, v: "text_embedding" not in k and "vision_embedding" not in k and 'text_embedding_bias' not in k and 'vision_embedding_bias' not in k}
                        state.pop("optimizer")
                        state.pop("epoch")
                    else:
                        filter = {"model": lambda k, v: "text_embedding" not in k and "vision_embedding" not in k}
                    self.fabric.save(save_path, state, filter=filter)
                    if self.fabric.global_rank == 0:
                        self.logger.info(set_color(f'save weight to {save_path}', color='cyan'))
                if self.early_stop is not None:
                    self.early_stop(val_score)
            elif self.early_stop is not None:
                self.early_stop(val_score)
                if self.fabric.global_rank == 0:
                    self.logger.info(set_color(f"The validation set score has not improved for {self.early_stop.counter} consecutive epochs. ", color="green"))

            if self.early_stop is not None and self.early_stop.early_stop:
                if self.fabric.global_rank == 0 and self.return_scores:
                    self.logger.info(set_color(f"Trigger early stop, the Epoch {self.best_epoch} is the best epoch of validation results, best_val_score is {self.best_val_score}.", color="cyan"))
                break
            if self.fabric.global_rank == 0:
                if self.return_scores:
                    log_dict = {'epoch': epoch + 1 + self.start_epoch, 'loss': loss_mean} | metrics
                else:
                    log_dict = {'epoch': epoch + 1 + self.start_epoch, 'loss': loss_mean}
                self.fabric.log_dict(log_dict, step=epoch + 1 + self.start_epoch)

