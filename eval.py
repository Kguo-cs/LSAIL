import pytorch_lightning as pl
from evaluate.CloseLoop_callback import ClosedLoopEvaluate
from datetime import datetime
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_Module import module
import argparse
import yaml

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--model_name', type=str, default="lasil")
parser.add_argument('--local_rank', type=int,default=0)
parser.add_argument('--ckpt_path', type=str,default='')

args = parser.parse_args()

path="./config/" + args.model_name + ".yaml"

with open(path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

model = module(cfg)

logger = loggers.TensorBoardLogger(save_dir=cfg['log_dir'],
                                   name=cfg['exp_name'],
                                   version=datetime.now().strftime("%Y_%m_%d_%H_%M")
                                   )

print("log_dir:",logger.log_dir)

checkpoint_cb = ModelCheckpoint(dirpath=logger.log_dir,
                                save_top_k=3,
                                monitor='val/collision_rate',
                                filename='{global_step:02d}-{val_loss:.4f}',
                               )


trainer = pl.Trainer(fast_dev_run=True,
                     logger=logger,
                     accelerator="gpu",
                     devices=-1,
                     limit_val_batches=128,
                     strategy="ddp",
                     log_every_n_steps=1000,
                     max_epochs=100,
                     callbacks=[ClosedLoopEvaluate(cfg),checkpoint_cb])

trainer.test(model,ckpt_path=args.ckpt_path)