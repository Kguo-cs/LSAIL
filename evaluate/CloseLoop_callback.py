import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from .evaluator import MyClosedLoopEvaluator
import time

class ClosedLoopEvaluate(Callback):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    def setup(self, trainer: pl.Trainer, pl_module, stage=None):

        self.num_scenes_per_device = self.cfg["eval"]["num_scenes_to_unroll"]

        self.rollout_batchsize = self.cfg['eval']['batch_size']

        eval_dataset = pl_module.val_dataset
        print("eval val")


        visualizer=None#Visualizer()#

        self.my_evaluator = MyClosedLoopEvaluator(visualizer,verbose=False)

        if self.cfg["model_params"]["rl_baseline"]:
            from .baseline_simulator import ClosedLoopSimulator
        else:
            from .Pneuma_simulator import ClosedLoopSimulator

        self.sim_loop = ClosedLoopSimulator(
            eval_dataset,
            pl_module.model,
            self.cfg["model_params"],
            evaluator=self.my_evaluator
        )

        eval_size = np.ceil(len(eval_dataset.index_array) / trainer.num_devices).astype(int)

        self.scene_ids = list(range(pl_module.global_rank, eval_size, trainer.num_devices))

        self.eval_len=len(eval_dataset.start_index)-1

        self.epoch=0

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        if not self.my_evaluator.set_up:
            self.my_evaluator.setup(pl_module.device,pl_module.meta_manager)

        with torch.no_grad():
            if  trainer.sanity_checking:
                scene_ids = np.random.choice(self.scene_ids, 1, replace=False)
            else:
                scene_ids = np.random.choice(self.scene_ids, min(self.num_scenes_per_device,len(self.scene_ids)), replace=False)

            results = self.roll_sim(pl_module, scene_ids)

            if results["pos_rmse"] > 100:
                print(scene_ids,results["pos_rmse"])
                trainer.save_checkpoint("abnormal.ckpt")

            self.epoch+=1

            if  self.epoch%5==0:
                i=pl_module.global_rank%self.eval_len

                sim_outs = self.sim_loop.longterm_unroll(pl_module.device,i)

                self.my_evaluator.longterm_eval(sim_outs,i,pl_module.device)

                longterm_results = self.my_evaluator.longterm_validate()

                results = {**results, **longterm_results}

            pl_module.log_dict(pl_module.val_metrics(results), batch_size=len(scene_ids))

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        with torch.no_grad():
            self.my_evaluator.setup(pl_module.device,pl_module.meta_manager)

            self.evaluate_sumo = False

            if self.evaluate_sumo:
                from .sumo_short_term import eval_sumo_shortterm
                from .sumo_long_term import eval_sumo_longterm

                eval_sumo_longterm(self.my_evaluator,self.sim_loop.dataset,pl_module.device)

            scene_ids = np.random.choice(self.scene_ids, 32,   replace=False)

            # scene_ids=[20450]

            results = self.roll_sim(pl_module,  scene_ids)

            print(results)

            print("start long term")

            start_time=time.time()

            self.my_evaluator.reset()

            for i in range(5):
                edge_data = self.sim_loop.longterm_unroll(pl_module.device, i)

                self.my_evaluator.longterm_eval(edge_data, i,pl_module.device)

            longterm_results = self.my_evaluator.longterm_validate()

            print(time.time()-start_time)

            print(longterm_results)

            results = {**results, **longterm_results}

        pl_module.log_dict(pl_module.val_metrics(results), batch_size=len(scene_ids))

    def roll_sim(self,pl_module, scene_ids):

        scene_num = len(scene_ids)

        batch_num = np.ceil(scene_num / self.rollout_batchsize).astype(int)

        for i in range(batch_num):
            scenes_to_unroll = list(
                scene_ids[i * self.rollout_batchsize:min((i + 1) * self.rollout_batchsize, scene_num)])

            sim_outs = self.sim_loop.unroll(scenes_to_unroll, pl_module.device, rollout_time=1)

            self.my_evaluator.evaluate(sim_outs,pl_module.device)

        shortterm_results = self.my_evaluator.validate()


        return shortterm_results