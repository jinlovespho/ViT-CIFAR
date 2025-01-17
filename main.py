import argparse

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as pl
import warmup_scheduler
import numpy as np

from util.utils import get_model, get_dataset, get_experiment_name, get_criterion
from util.da import CutMix, MixUp

# JINLOVESPHO
import os
import matplotlib.pyplot as plt 

import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.loggers import CSVLogger

from torchprofile import profile_macs 
from torchsummary import summary 


parser = argparse.ArgumentParser()
parser.add_argument("--api-key", default=None, help="API Key for Comet.ml")
parser.add_argument("--dataset", default="c100", type=str, help="[c10, c100, svhn]")
parser.add_argument("--num-classes", default=100, type=int)
parser.add_argument("--model-name", default="vit", help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int, help="number of patches in one row(col). not size of patch")
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max_epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment",default=True, action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", default=True, action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int, help='transformer input patch embedding dimension')
parser.add_argument("--mlp-hidden", default=384, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--project_name", default="VisionTransformer")
parser.add_argument('--experiment_memo', default='memo')
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--save_dir', type=str, default='/mnt/ssd2/')
parser.add_argument('--logger', type=str, default='comet')
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision=32

if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")

# breakpoint()

train_ds, test_ds = get_dataset(args)

plt.imsave('./img1.png', train_ds.data[0])  # train_ds.data[0] : (32,32,3)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)



class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        # breakpoint()
        print("Training Started!!")
        
    def on_train_end(self, trainer, pl_module):
        print("Training Finished!!")


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        # breakpoint()
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.api_key is None

    def forward(self, x):
        # breakpoint()
        return self.model(x)

    def configure_optimizers(self):
        # breakpoint()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]
    
    def training_step(self, batch, batch_idx):
        # breakpoint()
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_= self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        
        # 실행 O
        else:
            out = self.model(img)               # self(img) 해도 동일 
            loss = self.criterion(out, label)

        # 실행 O
        # if not self.log_image_flag and not self.hparams.dry_run:
        #     self.log_image_flag = True
        #     self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("train_loss",loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # breakpoint()
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self, *args, **kwargs):
        # breakpoint()
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True)
        
    def on_validation_epoch_end(self, *arg, **kwargs):

        # save every 50 epochs
        if self.current_epoch % 50 == 0:
            save_path = f'{args.save_dir}/model_checkpoints/{args.experiment_memo}'      
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.state_dict()['epoch'] = self.current_epoch 
            torch.save(self.state_dict(), f'{save_path}/epoch_{self.current_epoch}.pth')
            
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True)

    # def _log_image(self, image):
    #     breakpoint()
    #     grid = torchvision.utils.make_grid(image, nrow=4)
    #     self.logger.experiment.log_image(grid.permute(1,2,0))
    #     print("[INFO] LOG IMAGE!!!")


if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    # experiment_name = args.experiment_memo    
        
    if args.logger == 'wandb':
        print('[WANDB Logger]')
        logger = WandbLogger(
            log_model=False,
            project=args.project_name,
            name=experiment_name,
            save_dir=f'{args.save_dir}'
        )
    
    elif args.logger == 'comet':
        print("[INFO] Log with Comet.ml!")
        logger = CometLogger(
            api_key=args.api_key,
            save_dir=f'{args.save_dir}/comet',
            project_name=args.project_name,
            experiment_name=experiment_name
        )
        
    else:
        print("[INFO] Log with CSV")
        logger = CSVLogger(
            save_dir=f'{args.save_dir}/drive',
            name=experiment_name
        )
        
    
    net = Net(args)
    # net = torch.compile(net,  mode='reduce-overhead')
    
    # calculate and log: #of param, flops, and args
    # summ = summary(net.model, input_size=(3,32,32), batch_size=1, device='cpu')
    # macs = profile_macs(net.model, torch.rand(1,3,32,32))
    # tot_param = sum( i.numel() for i in net.model.parameters() if i.requires_grad )

    # logger.log_hyperparams({'Params': tot_param, 'MACS':macs, 'FLOPs':macs*2 })
    # logger.log_hyperparams(args)

    # breakpoint()
    
    # trainer
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, accelerator='auto', benchmark=args.benchmark, 
                         logger=logger, max_epochs=args.max_epochs+1, enable_model_summary=True, callbacks=[PrintCallback()])
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)
    
    print(' End of Main')
