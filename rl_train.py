from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)

import ray
import logging

torch, nn = try_import_torch()
from ray import air, tune
from environment import Env
from callbacks import MultiAgentDrivingCallbacks, progress_reporter
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)


class GAILConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or GAIL)

        # self.rollout_fragment_length = 400#train_batchsize/rollout_worker
        self.train_batch_size = 5
        # we subsample a batch of `sgd_minibatch_size` from the train-batch for
        # each `num_sgd_iter`).
        self.sgd_minibatch_size = min(512, self.train_batch_size)  # the train batch is be split into 512 chunks, each of which is iterated over (used for updating the policy) 5 times.

        self.lr = 3e-4
        self.clip_param = 0.2
        self.lambda_ = 0.95

        self.num_cpus_per_worker = 1
        self.num_cpus_for_local_worker = 1

        self.framework_str = "torch"

        self.num_sgd_iter = 1
        self.num_rollout_workers = 1
        # Two important updates
        self.vf_clip_param = 100
        self.old_value_loss = True

        self.env = Env

        self.update_from_dict({"model": {"custom_model": "rl_model"}})  # ,"use_airl": True

        self.freq = 10

        self.gail_reward = True

        self.only_gailreward=True

        self.ori_loss = True

        self.airl = False  # reward should be   D=exp(f)/(exp(f)+pi) f=g(s,a)+gamma*h(s')-h(s)

        self.bc_w = 0

        self.gp=0

        self.vae=True

class IPPOPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):

        env_config = config["env_config"]

        env_config["std"] = config["bcstd"] if "bcstd" in config else 0

        self.data = Env(env_config)

        self.iter = 0

        super(IPPOPolicy, self).__init__(observation_space, action_space, config)

    def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            if self.config["gail_reward"]:
                if 'actions' in sample_batch.keys():

                    infos = sample_batch['infos']

                    if infos.dtype != 'float32':
                        if 'actions' in sample_batch.keys():
                            obs = sample_batch['obs']

                            action = sample_batch['actions']

                            rewards=sample_batch["rewards"]

                            label_pred = self.model.reward(obs, action)

                            gail_rewards = -torch.log(label_pred + 1e-5)

                            gail_rewards = gail_rewards.cpu().numpy()

                            if self.config["only_gailreward"]:
                                rewards=gail_rewards
                            else:
                                rewards+=gail_rewards

                            sample_batch['rewards'] = rewards
                            for reward, info in zip(rewards, infos):
                                info['episode_reward'] = reward
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

    def loss(self, model, dist_class, train_batch):

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid
        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] *
            torch.clamp(logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]),
        )

        # Compute a value function loss.
        assert self.config["use_critic"]

        value_fn_out = model.value_function()

        if self.config["old_value_loss"]:
            current_vf = value_fn_out
            prev_vf = train_batch[SampleBatch.VF_PREDS]
            vf_loss1 = torch.pow(current_vf - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_clipped = prev_vf + torch.clamp(
                current_vf - prev_vf, -self.config["vf_clip_param"], self.config["vf_clip_param"]
            )
            vf_loss2 = torch.pow(vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.max(vf_loss1, vf_loss2)
        else:
            vf_loss = torch.pow(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)

        total_loss = reduce_mean_valid(
            -surrogate_loss + self.config["vf_loss_coeff"] * vf_loss_clipped - self.entropy_coeff * curr_entropy
        )

        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        if self.config['gail_reward'] ==False and self.config["only_gailreward"]:
            expert_obs, expert_actions, expert_dones = self.data.sample()
            model_out = model.logit(expert_obs)
            action_dist = dist_class(model_out, model)
            actions = torch.tensor(expert_actions).to(model_out.device)
            # log\pi_\theta(a|s)
            logprobs = action_dist.logp(actions)

            p_loss = -logprobs.mean()

            bc_weight=1#self.config['bc_w']

            total_loss = total_loss+p_loss * bc_weight
        else:
            expert_obs=None

        if self.config["gail_reward"] and self.iter % self.config['freq'] == 0:
            if expert_obs is None:

                expert_obs, expert_actions, expert_dones = self.data.sample()

            policy_obs = train_batch["obs"]

            policy_actions = train_batch['actions']

            expert_pred = model.reward(expert_obs, expert_actions)  # expert 0 policy 1

            policy_pred = model.reward(policy_obs, policy_actions)

            expert_loss = -torch.log((1 - expert_pred)).mean()  # +1e-3

            policy_loss = -torch.log(policy_pred).mean()  # +1e-3

            total_loss += expert_loss + policy_loss

        self.iter += 1

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss


class GAIL(PPO):
    @classmethod
    def get_default_config(cls):
        return GAILConfig()

    def get_default_policy_class(self, config):
        return IPPOPolicy

from ray.rllib.models.catalog import MODEL_DEFAULTS,ModelCatalog

MODEL_DEFAULTS["fcnet_hiddens"]=[512, 512,512,256]

from rl_model import RL_Model

ModelCatalog.register_custom_model("rl_model", RL_Model)

config = (
        GAILConfig()
        .update_from_dict(dict(sgd_minibatch_size=150, train_batch_size=150, num_sgd_iter=5,gail_reward=tune.grid_search([True,False]),only_gailreward=tune.grid_search([False,True]) ))
        .environment(env_config=dict(type="train", fut_dim=tune.grid_search([10]), std=tune.grid_search([0]), post=tune.grid_search([True])))
        .framework('torch')
        .resources(num_gpus=1)
        .callbacks(MultiAgentDrivingCallbacks)
        .rollouts(num_rollout_workers=4)
        .evaluation(
        evaluation_num_workers=1,
        evaluation_interval=1,
        evaluation_duration='auto',
        evaluation_parallel_to_training=True,
        evaluation_config={"env_config": dict(type="val", std=0,iter_n=20)})
)

# evaluation will do the same thing
debug = False#You are running ray with `local_mode=True`, but have configured 1 GPUs to be used!

ray.init(
    logging_level=logging.ERROR,
    log_to_driver=False,
    local_mode=debug,
    num_gpus=torch.cuda.device_count(),
    ignore_reinit_error=True,
)

print("Available resources: ", ray.available_resources())

from ray.air.config import CheckpointConfig

def trail_name(trail):
    config = trail.config['env_config']

    fut_dim = config["fut_dim"] if "fut_dim" in config else 1
    std = config["std"] if "std" in config else 0
    post = config["post"] if "post" in config else 0

    return "fut_" + str(fut_dim) + "std_" + str(std) + "post_" + str(post)


checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_order='min', checkpoint_score_attribute='minade_val',
                                     checkpoint_frequency=1)

tuner = tune.Tuner(
    GAIL,
    param_space=config.to_dict(),
    run_config=air.RunConfig(local_dir='./ray_exp', verbose=1, progress_reporter=progress_reporter,
                             checkpoint_config=checkpoint_config),
    # tune_config=TuneConfig(trial_name_creator=trail_name,trial_dirname_creator=trail_name)
)
results = tuner.fit()
