import os
import torch
import logging
import argparse
import configparser
import lightning as L
import model_utils
import warnings

from functools import partial
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Union

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from model_utils import get_queries, eval_batch, construct_lora_vggt, get_points_on_a_grid
from models.cotracker import Cotracker

from data.kubric_movif_dataset import KubricMovifDataset
from data.panoptic_dataset import PanopticDataset
from data.utils import collate_fn

warnings.filterwarnings("ignore", message="No device id is provided", module="torch.distributed")
torch.set_float32_matmul_precision('high')

class CotrackerModel(L.LightningModule):
    def __init__(
        self,
        max_steps: int = 100000,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_forward_kwargs: Optional[Dict[str, Any]] = None,
        loss_name: Optional[str] = 'tapir_loss',
        eval_loss_name: Optional[str] = 'tapir_loss',
        loss_kwargs: Optional[Dict[str, Any]] = None,
        query_first: Optional[bool] = False,          
        optimizer_name: Optional[str] = 'Adam',
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_name: Optional[str] = 'OneCycleLR',
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        lora_usage: bool = False,
        lora_kwargs: Optional[Dict[str, Any]] = None,
        
    ):
        super().__init__()
        self.model = Cotracker(**(model_kwargs or {}))

        self.model_forward_kwargs = model_forward_kwargs or {}
        self.loss = partial(model_utils.__dict__[loss_name], **(loss_kwargs or {}))
        self.eval_loss = partial(model_utils.__dict__[eval_loss_name], **(loss_kwargs or {}))
        self.query_first = query_first

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 2e-3}
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs or {'max_lr': 2e-3, 'pct_start': 0.05, 'total_steps': 300000}
        self.scheduler_kwargs['max_lr'] = self.optimizer_kwargs['lr']
        self.scheduler_kwargs['total_steps'] = max_steps + 100  

        self.automatic_optimization=False

        self.lora_usage = lora_usage
        

        if self.lora_usage:
            construct_lora_vggt(self.model, **(lora_kwargs or {}))

        
    def training_step(self, batch, batch_idx):
        video = batch['video']
        trajs_g = batch['trajectory']
        vis_g = batch['visibility']
        valids = batch['valid']

        queries = get_queries(trajs_g, vis_g, video.device)
        if queries is None:
            valids = torch.zeros_like(valids).to(valids.device).float()

        output = self.model(video, queries, **self.model_forward_kwargs)

        loss, loss_scalars = self.loss(batch, output, self.model.window_len)

        self.log_dict(
            {f'train/{k}': v.item() for k, v in loss_scalars.items()},
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        opt = self.optimizers()
        sched = self.lr_schedulers()
        opt.zero_grad()
        self.manual_backward(loss)

        if any([p.grad.isnan().any() for p in self.parameters() if p.grad is not None]):
            print('nan gradients detected, skipping step')
            [p.grad.zero_() for p in self.parameters() if p.grad is not None and p.grad.isnan().any()]

        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

        opt.step()
        sched.step()

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
       
        video = batch['video']
        queries = batch['query_points'].clone().float()


        queries = torch.stack(
            [
                queries[:, :, 0],
                queries[:, :, 2],
                queries[:, :, 1],
            ],
            dim=2,
        )

        # get support grid
        xy = get_points_on_a_grid(5, video.shape[3:])
        xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(queries.device)
        queries = torch.cat([queries, xy], dim=1)


        output = self.model(video, queries, **self.model_forward_kwargs)

        # _, loss_scalars = self.loss(batch, output, self.model.window_len)

        metrics = eval_batch(batch, output, query_first=self.query_first)

        log_prefix = 'val/'
        if dataloader_idx is not None:
            log_prefix = f'val/data_{dataloader_idx}/'

        # self.log_dict(
        #     {log_prefix + k: v for k, v in loss_scalars.items()},
        #     logger=True,
        #     sync_dist=True,
        # )
        self.log_dict(
            {log_prefix + k: v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
        )
        logging.info(f"Batch {batch_idx}: {metrics}")

    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        
        video = batch['video']
        queries = batch['query_points'].clone().float()


        queries = torch.stack(
            [
                queries[:, :, 0],
                queries[:, :, 2],
                queries[:, :, 1],
            ],
            dim=2,
        )

        # get support grid
        xy = get_points_on_a_grid(5, video.shape[3:])
        xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2)
        queries = torch.cat([queries, xy], dim=1)

        output = self.model(video, queries, **self.model_forward_kwargs)

        # _, loss_scalars = self.loss(batch, output, self.model.window_len)

        metrics = eval_batch(batch, output, query_first=self.query_first)

        log_prefix = 'test/'
        if dataloader_idx is not None:
            log_prefix = f'test/data_{dataloader_idx}/'

        # self.log_dict(
        #     {log_prefix + k: v for k, v in loss_scalars.items()},
        #     logger=True,
        #     sync_dist=True,
        # )
        self.log_dict(
            {log_prefix + k: v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
        )
        logging.info(f"Batch {batch_idx}: {metrics}")

    def configure_optimizers(self):
        
        base_lr = self.optimizer_kwargs.get("lr", 2e-3)
        base_wd = self.optimizer_kwargs.get("wdecay", 1e-4)
        eps = self.optimizer_kwargs.get("eps", 1e-8)


        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Param Count] Total trainable parameters: {total_params:,}")

        
        Optim = getattr(torch.optim, self.optimizer_name)
        optimizer = Optim(trainable_params, lr=base_lr, weight_decay=base_wd, eps=eps)

        
        sch_kwargs = dict(self.scheduler_kwargs)
        
        sch_kwargs["max_lr"] = base_lr
        Sched = getattr(torch.optim.lr_scheduler, self.scheduler_name)
        scheduler = Sched(optimizer, **sch_kwargs)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    
    # gpt-5
    # def configure_optimizers(self):
    #     base_lr = self.optimizer_kwargs.get("lr", 2e-3)
    #     base_wd = self.optimizer_kwargs.get("wdecay", 1e-4)
    #     eps     = self.optimizer_kwargs.get("eps", 1e-8)

    #     non_fnet, lora, pos = [], [], None
    #     for name, p in self.model.named_parameters():
    #         if not p.requires_grad: 
    #             continue
    #         if 'fnet' not in name:
    #             non_fnet.append(p)
    #         else:
    #             if 'lora_' in name or 'lora_A' in name or 'lora_B' in name:
    #                 lora.append(p)

    #     try:
    #         pe = self.model.fnet.patch_embed.pos_embed
    #         if pe.requires_grad:
    #             pos = pe
    #     except: pass


    #     non_fnet_count = sum(p.numel() for p in non_fnet)
    #     lora_count     = sum(p.numel() for p in lora)
    #     pos_count      = pos.numel() if pos is not None else 0
    #     total_count    = non_fnet_count + lora_count + pos_count

    #     print(f"[Param Count] non_fnet: {non_fnet_count:,}")
    #     print(f"[Param Count] lora:     {lora_count:,}")
    #     print(f"[Param Count] pos:      {pos_count:,}")
    #     print(f"[Param Count] total:    {total_count:,}")

    #     groups = []
    #     if non_fnet: groups.append({"params": non_fnet, "lr": base_lr, "weight_decay": base_wd})
    #     if lora:
    #         mult = self.optimizer_kwargs.get("lora_lr_mult", 5.0)
    #         groups.append({"params": lora, "lr": base_lr*mult, "weight_decay": 0.0})
    #     if pos is not None:
    #         groups.append({"params": [pos], "lr": base_lr*0.1, "weight_decay": 0.0})

    #     Optim = getattr(torch.optim, self.optimizer_name)
    #     optimizer = Optim(groups if groups else [{"params": self.parameters()}],
    #                     lr=base_lr, weight_decay=base_wd, eps=eps)

        
    #     max_lrs = [g.get("lr", base_lr) for g in groups] or base_lr
    #     sch_kwargs = dict(self.scheduler_kwargs)
    #     sch_kwargs["max_lr"] = max_lrs
    #     Sched = getattr(torch.optim.lr_scheduler, self.scheduler_name)
    #     scheduler = Sched(optimizer, **sch_kwargs)
    #     return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    


def train(
    mode: str,
    save_path: str,
    val_dataset_path: str,
    ckpt_path: str = None,
    kubric_dir: str = '',
    precision: str = '32',
    batch_size: int = 1,
    crop_size: int = 256,
    seq_len : int = 24,
    traj_per_sample : int = 224,
    use_augs: bool = False,
    vggt_size: int = 224,
    val_check_interval: Union[int, float] = 5000,
    log_every_n_steps: int = 10,
    gradient_clip_val: float = 1.0,
    max_steps: int = 300_000,    
    lora_usage: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    model_forward_kwargs: Optional[Dict[str, Any]] = None,
    lora_kwargs: Optional[Dict[str, Any]] = None,
    loss_name: Optional[str] = 'tapir_loss',
    eval_loss_name: Optional[str] = 'tapir_loss',
    loss_kwargs: Optional[Dict[str, Any]] = None,
    query_first: Optional[bool] = True,
    optimizer_name: Optional[str] = 'Adam',
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    scheduler_name: Optional[str] = 'OneCycleLR',
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
):
    
    seed_everything(42, workers=True)


    model = CotrackerModel(
        model_kwargs=model_kwargs,
        model_forward_kwargs=model_forward_kwargs,
        loss_name=loss_name,
        eval_loss_name=eval_loss_name,
        loss_kwargs=loss_kwargs,
        query_first=query_first,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
        lora_usage=lora_usage,
        lora_kwargs=lora_kwargs
    )

    if ckpt_path is not None and 'train' in mode:
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    
    logger = WandbLogger(project='Cotracker', save_dir=save_path, id=os.path.basename(save_path))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_last=True,
        save_top_k=3,
        mode="max",
        monitor="val/average_pts_within_thresh",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
    )

    eval_dataset = PanopticDataset(
            data_root=val_dataset_path,
            dataset_type='panoptic',
            resize_to=vggt_size,
    )


    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    if 'train' in mode:
        trainer = L.Trainer(
            strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
            logger=logger,
            precision=precision,
            val_check_interval=val_check_interval,
            log_every_n_steps=log_every_n_steps,
            max_steps=max_steps,
            sync_batchnorm=True,
            callbacks=[checkpoint_callback, lr_monitor],
        )

        train_ds = KubricMovifDataset(
            data_root=kubric_dir,
            crop_size=(crop_size, crop_size),
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            use_augs=use_augs,
            vggt_size=(vggt_size, vggt_size),
        )


        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,     
            shuffle=True,              
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=collate_fn,  
        )

        trainer.fit(model, train_loader, eval_dataloader, ckpt_path=ckpt_path)
        # trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    elif 'eval' in mode:
        trainer = L.Trainer(strategy='ddp', logger=logger, precision=precision)
        trainer.test(model, eval_dataloader, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Invalid mode: {mode}")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate the LocoTrack model.")
    parser.add_argument('--config', type=str, default='config/train.ini', help="Path to the configuration file.")
    parser.add_argument('--mode', type=str, required=True, help="Mode to run: 'train' or 'eval' with optional 'q_first' and the name of evaluation dataset.")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument('--save_path', type=str, default='snapshots', help="Path to save the logs and checkpoints.")
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Extract parameters from the config file
    train_params = {
        'mode' : args.mode,
        'ckpt_path' : args.ckpt_path,
        'save_path' : args.save_path,
        'val_dataset_path': eval(config.get('TRAINING', 'val_dataset_path', fallback='{}')),
        'kubric_dir': config.get('TRAINING', 'kubric_dir', fallback=''),
        'precision': config.get('TRAINING', 'precision', fallback='32'),
        'lora_usage': config.getboolean('TRAINING', 'lora_usage', fallback=True),
        'batch_size': config.getint('TRAINING', 'batch_size', fallback=1),
        'crop_size': config.getint('TRAINING', 'crop_size', fallback=256),
        'seq_len': config.getint('TRAINING', 'seq_len', fallback=24),
        'traj_per_sample': config.getint('TRAINING', 'traj_per_sample', fallback=224),
        'use_augs': config.getboolean('TRAINING', 'use_augs', fallback=False),
        'vggt_size': config.getint('TRAINING', 'vggt_size', fallback=224),
        'val_check_interval': config.getfloat('TRAINING', 'val_check_interval', fallback=5000),
        'log_every_n_steps': config.getint('TRAINING', 'log_every_n_steps', fallback=10),
        'gradient_clip_val': config.getfloat('TRAINING', 'gradient_clip_val', fallback=1.0),
        'max_steps': config.getint('TRAINING', 'max_steps', fallback=300000),
        'query_first': config.getboolean('TRAINING', 'query_first', fallback=True),
        'model_kwargs': eval(config.get('MODEL', 'model_kwargs', fallback='{}')),
        'model_forward_kwargs': eval(config.get('MODEL', 'model_forward_kwargs', fallback='{}')),
        'lora_kwargs': eval(config.get('LORA', 'lora_kwargs', fallback='{}')),
        'loss_name': config.get('LOSS', 'loss_name', fallback='tapir_loss'),
        'eval_loss_name': config.get('LOSS', 'eval_loss_name', fallback='tapir_loss'),
        'loss_kwargs': eval(config.get('LOSS', 'loss_kwargs', fallback='{}')),
        'optimizer_name': config.get('OPTIMIZER', 'optimizer_name', fallback='Adam'),
        'optimizer_kwargs': eval(config.get('OPTIMIZER', 'optimizer_kwargs', fallback='{"lr": 2e-3}')),
        'scheduler_name': config.get('SCHEDULER', 'scheduler_name', fallback='OneCycleLR'),
        'scheduler_kwargs': eval(config.get('SCHEDULER', 'scheduler_kwargs', fallback='{"max_lr": 2e-3, "pct_start": 0.05, "total_steps": 300000}')),
    }

    train(**train_params)
