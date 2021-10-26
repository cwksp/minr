"""
    run()
        make_datasets()
            . train_loader, test_loader, dist_samplers
        make_model()
            . model_ddp, model
        train()
            . optimizer, epoch, log_buffer
            for range(max_epoch):
                adjust_learning_rate()
                train_epoch()
                    train_step()
                evaluate_epoch()
                    evaluate_step()
                visualize_epoch()
                save_checkpoint()
"""

import os
import os.path as osp
import time

import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import datasets
import models
import utils


class BaseTrainer():

    def __init__(self, rank, cfg):
        self.cfg = cfg
        self.rank = rank
        self.is_master = (rank == 0)
        self.tot_gpus = cfg['tot_gpus']
        self.distributed = (cfg['tot_gpus'] > 1)
        self.num_workers = cfg['num_workers'] // cfg['tot_gpus']

        # Set logs
        if self.is_master:
            logger, writer = utils.set_save_dir(cfg['save_dir'], replace=False)
            with open(osp.join(cfg['save_dir'], 'cfg.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)
            self.log = logger.info

            self.enable_tb = True
            self.writer = writer

            if cfg.get('wandb_upload', False):
                self.enable_wandb = True
                self.setup_wandb()
            else:
                self.enable_wandb = False
        else:
            def empty_fn(*args, **kwargs):
                pass
            self.log = empty_fn
            self.enable_tb = False
            self.enable_wandb = False

        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', torch.cuda.current_device())

        if self.distributed:
            dist_url = f"tcp://localhost:{cfg['port']}"
            dist.init_process_group(backend='nccl', init_method=dist_url,
                                    world_size=self.tot_gpus, rank=rank)
            self.log(f'Distributed training enabled.')

        cudnn.benchmark = cfg.get('cudnn', False)
        self.log(f'Environment setup done.')

    def setup_wandb(self):
        cfg = self.cfg
        os.environ['WANDB_NAME'] = cfg['exp_name']
        os.environ['WANDB_DIR'] = cfg['save_dir']
        if not cfg.get('wandb_upload', False):
            os.environ['WANDB_MODE'] = 'dryrun'
        os.environ['WANDB_API_KEY'] = cfg['wandb']['api_key']
        wandb.init(project=cfg['wandb']['project'], entity=cfg['wandb']['entity'], config=cfg)

    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=t)
        if self.enable_wandb:
            wandb.log({k: v}, step=t)

    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.tot_gpus)

    def sync_ave_scalars_(self, ave_scalars):
        for k in ave_scalars.keys():
            x = torch.tensor([ave_scalars[k].item()], device=self.device)
            self.dist_all_reduce_mean_(x)
            ave_scalars[k].v = x.mean().item()
            ave_scalars[k].n *= self.tot_gpus

    def make_datasets(self):
        cfg = self.cfg
        self.dist_samplers = []

        def make_distributed_loader(dataset, batch_size, shuffle=False, drop_last=False):
            sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
            loader = DataLoader(
                dataset, batch_size // self.tot_gpus, drop_last=drop_last,
                sampler=sampler, shuffle=((sampler is None) and shuffle),
                num_workers=self.num_workers, pin_memory=True)
            return loader, sampler

        if cfg.get('train_dataset') is not None:
            train_dataset = datasets.make(cfg['train_dataset'])
            self.log(f'Train dataset: len={len(train_dataset)}')
            self.train_loader, train_sampler = make_distributed_loader(train_dataset, cfg['batch_size'], shuffle=True, drop_last=True)
            self.dist_samplers.append(train_sampler)

        if cfg.get('test_dataset') is not None:
            test_dataset = datasets.make(cfg['test_dataset'])
            self.log(f'Test dataset: len={len(test_dataset)}')
            self.test_loader, test_sampler = make_distributed_loader(test_dataset, cfg['batch_size'], shuffle=False, drop_last=False)
            self.dist_samplers.append(test_sampler)

    def make_model(self, model_spec=None):
        if model_spec is None:
            model_spec = self.cfg['model']
        load_sd = (model_spec.get('sd') is not None)
        model = models.make(model_spec, load_sd=load_sd)
        self.log(f'Model: #params={utils.compute_num_params(model)}')

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(model, device_ids=[self.rank])
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp

    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

    def train_step(self, data):
        raise NotImplementedError

    def train_epoch(self):
        self.model_ddp.train()
        ave_scalars = dict()

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='train', leave=False)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.train_step(data)
            self.t_model += time.time() - t0

            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, 1)

            if self.is_master:
                pbar.set_description(desc=f'train: loss={ret["loss"]:.4f}')
            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'train:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('train/' + k, v.item())
        self.log_buffer.append(logtext)

    def evaluate_step(self, data):
        raise NotImplementedError

    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        pbar = self.test_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='eval', leave=False)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.evaluate_step(data)
            self.t_model += time.time() - t0

            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, 1)

            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'eval:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('test/' + k, v.item())
        self.log_buffer.append(logtext)

    def visualize_epoch(self):
        pass

    def save_checkpoint(self, filename='epoch-last.pth'):
        if not self.is_master:
            return
        model_spec = self.cfg['model']
        model_spec['sd'] = self.model.state_dict()
        optimizer_spec = self.cfg['optimizer']
        optimizer_spec['sd'] = self.optimizer.state_dict()
        checkpoint = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': self.epoch,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, osp.join(self.cfg['save_dir'], filename))

    def train(self):
        cfg = self.cfg

        self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), cfg['optimizer'])

        max_epoch = cfg['max_epoch']
        eval_epoch = cfg.get('eval_epoch', max_epoch)
        vis_epoch = cfg.get('vis_epoch', eval_epoch)
        save_epoch = cfg.get('save_epoch', max_epoch + 1)
        epoch_timer = utils.EpochTimer(max_epoch)

        for epoch in range(1, max_epoch + 1):
            self.epoch = epoch
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)

            self.adjust_learning_rate()
            self.log_temp_scalar('lr', self.optimizer.param_groups[0]['lr'])

            self.t_data, self.t_model = 0, 0
            self.train_epoch()

            if epoch % eval_epoch == 0:
                self.evaluate_epoch()
            if epoch % vis_epoch == 0:
                self.visualize_epoch()

            if epoch % save_epoch == 0:
                self.save_checkpoint(f'epoch-{epoch}.pth')
            self.save_checkpoint('epoch-last.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_step()
            data_ratio = self.t_data / (self.t_data + self.t_model)
            self.log_buffer.append('{} (d {:.2f}) {}/{}'.format(epoch_time, data_ratio, tot_time, est_time))
            self.log(', '.join(self.log_buffer))

            from IPython import embed; embed()

    def run(self):
        self.make_datasets()

        if self.cfg.get('eval_model') is None:
            self.make_model()
            self.train()
        else:
            model_spec = torch.load(self.cfg['eval_model'])['model']
            self.make_model(model_spec)
            self.epoch = 0
            self.log_buffer = []
            self.evaluate_epoch()
            self.visualize_epoch()
            self.log(', '.join(self.log_buffer))

        if self.is_master:
            if self.enable_tb:
                self.writer.close()
            if self.enable_wandb:
                wandb.finish()
