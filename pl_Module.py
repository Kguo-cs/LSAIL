import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
import torch
from data.Pneuma_dataset import PneumaDataset
from data.Pneuma_manager import PneumaManager
import os
from data.feature_collate import FeatureCollate
import numpy as np
from evaluate.Pneuma_simulator import ClosedLoopSimulator
from model.model import Model
from data.rollout_buffer import RolloutBuffer
from evaluate.metric_utils import DictMetric,val_metrics,train_metrics
import pickle
#from static_vis import vis

class module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg=cfg
        model_cfg=cfg['model_params']

        rl_baseline=model_cfg['rl_baseline']

        self.dataset_setup()
        self.val_metrics = DictMetric(val_metrics, prefix="val/")

        self.train_metrics = DictMetric(train_metrics, prefix="train/")

        if rl_baseline:
            from rl_model import RL_Model
            import gym

            model_config={'_use_default_native_models': False, '_disable_preprocessor_api': False, '_disable_action_flattening': False, 'fcnet_hiddens': [512, 512, 512, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': 'my_model', 'custom_model_config': {}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}

            num_outputs=4*model_cfg["future_num_frames"]
            act_dim = model_cfg["future_num_frames"] * 2

            obs_space=gym.spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(262,),
                dtype=np.float32
            )

            action_space= gym.spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(act_dim,),
                dtype=np.float32
            )

            self.model = RL_Model(obs_space, action_space, num_outputs, model_config, name="model")

            path = "./baseline/"+cfg["exp_name"]+".pkl"
            with open(path, "rb") as f:
                state = pickle.load(f)

            weights = state['weights']

            for key, item in weights.items():
                weights[key] = torch.tensor(item)

            self.model.load_state_dict(weights)

        else:
            self.model = Model(model_cfg)

            self.learner_w = model_cfg["learner_w"]

            self.output_vel=model_cfg["output_vel"]

            self.output_gaussian=model_cfg["output_gaussian"]

            self.l1loss = nn.L1Loss(reduction="mean")

            if self.learner_w:

                self.rollout_buffer=RolloutBuffer()

                self.sim_loop = ClosedLoopSimulator(self.train_dataset,self.model,model_cfg,buffer=self.rollout_buffer)

                self.scene_ids = list(range( len(self.train_dataset.index_array)))

                self.rollout_len=model_cfg["rollout_len"]

                self.rollout_interval=model_cfg["rollout_interval"]

                self.train_batch_size = cfg['train_data_loader']['batch_size']

            self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        output = {}

        expert_state=batch["agent_polylines"]

        expert_action=batch["positions"]

        if  self.learner_w!=0 and self.global_step%self.rollout_interval==0 :
            with torch.no_grad():
                scenes_to_unroll = np.random.choice(self.scene_ids, self.train_batch_size, replace=False)
                self.model.eval()
                self.sim_loop.unroll(scenes_to_unroll, self.device,sim_len=self.rollout_len, training=True)
                self.model.train()

        pred_positions, ori_pos,rotate,  expert_vae_loss,expert_mask = self.model(expert_state)

        expert_fut = expert_action[expert_mask]

        expert_fut_mask = expert_fut[..., -1] != 0

        rel_expert_fut = expert_fut[..., :2] - ori_pos

        rel_expert_fut = torch.einsum("nta,nab->ntb", rel_expert_fut, rotate)

        if self.output_vel:
            vel=rel_expert_fut[:,1:]-rel_expert_fut[:,:-1]

            rel_expert_fut=torch.cat([rel_expert_fut[:,:1],vel],dim=1)

            expert_fut_mask[:,1:]&=expert_fut_mask[:,:-1]

        if self.output_gaussian:

            curr_dist = self.model.parameterize_gaussian(pred_positions[expert_fut_mask])

            neg_logp = -curr_dist.log_prob(rel_expert_fut[expert_fut_mask])

            bc_loss=neg_logp.mean()
        else:
            bc_loss=self.l1loss(pred_positions[expert_fut_mask],rel_expert_fut[expert_fut_mask])

        policy_loss = bc_loss+ expert_vae_loss[0]+expert_vae_loss[1]


        if self.learner_w!=0:

            learner_state_info = self.rollout_buffer.sample_state()

            learner_graph_state,  learner_rel_relation = learner_state_info

            learner_vae_loss = self.model.vae_model.vae_rec_hist(learner_graph_state,rel_relation=learner_rel_relation)[0]

            policy_loss +=(learner_vae_loss[0]+learner_vae_loss[1] )* self.learner_w#

            output["learner_rec_loss"] = learner_vae_loss[0]
            output["learner_kl_loss"] = learner_vae_loss[1]

        output["loss"] = policy_loss
        output["bc_loss"] = bc_loss
        output["expert_rec_loss"] = expert_vae_loss[0]
        output["expert_kl_loss"] = expert_vae_loss[1]

        return output

    def training_step_end(self,output):

        self.log_dict(self.train_metrics(output), batch_size=1)

    def validation_step(self, batch, batch_idx):
        output={"x_diff_mean":0,
                "y_diff_mean":0
                }
        return output

    def validation_step_end(self,output):

        self.log_dict(self.val_metrics(output), batch_size=1)

    def test_step(self, batch, batch_idx):

        return self.validation_step(batch,batch_idx)

    def test_step_end(self, output):

        self.log_dict(self.val_metrics(output),batch_size=1)

    def dataset_setup(self):
        self.num_workers = len(os.sched_getaffinity(0)) // torch.cuda.device_count()

        self.meta_manager = PneumaManager(self.cfg)

        self.train_dataset = PneumaDataset(self.cfg, 'train', self.meta_manager)

        self.val_dataset = PneumaDataset(self.cfg, 'val', self.meta_manager)

    def build_dataloader(self,dataset,data_loader_cfg):

        data_loader = DataLoader(dataset,
                                  shuffle=data_loader_cfg["shuffle"],
                                  batch_size=data_loader_cfg["batch_size"],
                                  num_workers=self.num_workers,
                                  prefetch_factor=2,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=FeatureCollate()
                                 )
        return data_loader

    def train_dataloader(self):
        train_cfg = self.cfg['train_data_loader']

        return self.build_dataloader(self.train_dataset,train_cfg)


    def val_dataloader(self):
        val_cfg = self.cfg['val_data_loader']

        return self.build_dataloader(self.val_dataset,val_cfg)

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):

        policy_optimizer= torch.optim.Adam(self.model.parameters(), lr=3e-4,eps=1e-5)

        optim= {'optimizer': policy_optimizer}

        return optim