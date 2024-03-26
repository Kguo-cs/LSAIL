import pytorch_lightning as pl
import yaml
from evaluate.CloseLoop_callback import ClosedLoopEvaluate
from datetime import datetime
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_Module import module
import argparse

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--model_name', type=str,   default="lasil")
parser.add_argument('--local_rank', type=int,  default=0)

args = parser.parse_args()

path="./config/" + args.model_name + ".yaml"

with open(path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

module = module(cfg)

logger = loggers.TensorBoardLogger(save_dir=cfg['log_dir'],
                                   name=cfg['exp_name'],
                                   version=datetime.now().strftime("%Y_%m_%d_%H_%M")
                                   )

print("log_dir:",logger.log_dir)

checkpoint_cb = ModelCheckpoint(dirpath=logger.log_dir,
                                save_top_k=100,
                                monitor='val/pos_rmse',
                                filename='{epoch}-{step}',
                                )

val_check_interval=cfg["eval"]["val_check_interval"]

if val_check_interval==0:
    val_check_interval=None

trainer = pl.Trainer(fast_dev_run=False,
                     logger=logger,
                     accelerator="gpu",
                     devices=-1,
                     limit_val_batches=1,
                     val_check_interval=val_check_interval,
                     strategy="ddp",
                     log_every_n_steps=500,
                     max_epochs=1000,
                     callbacks=[ClosedLoopEvaluate(cfg),checkpoint_cb])

trainer.fit(module)