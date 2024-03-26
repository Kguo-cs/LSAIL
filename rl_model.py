from typing import Type

import gym
import numpy as np
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TrainerConfigDict
torch, nn = try_import_torch()



class RL_Model(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""
    def __init__(
        self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
        model_config: ModelConfigDict, name: str
    ):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = False#model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None


        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        self._hidden_layers = nn.Sequential(*layers)


        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.

            # ========== Our Modification ==========
            # Note: We use centralized critic obs size as the input size of critic!
            # prev_vf_layer_size = int(np.product(obs_space.shape))
            prev_vf_layer_size = int(np.product(obs_space.shape))
            assert prev_vf_layer_size > 0

            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_vf_layer_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None
        )


        prev_reward_layer_size = int(np.product(obs_space.shape))+int(np.product(action_space.shape))

        reward__layers = []
        for size in hiddens:
            reward__layers.append(
                SlimFC(
                    in_size=prev_reward_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)
                )
            )
            prev_reward_layer_size = size
        self._reward_branch_separate = nn.Sequential(*reward__layers)

        self._reward_branch = SlimFC(
            in_size=prev_reward_layer_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None
        )


        prev_reward_layer_size1 = int(np.product(obs_space.shape))

        reward__layers = []
        for size in hiddens:
            reward__layers.append(
                SlimFC(
                    in_size=prev_reward_layer_size1,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)
                )
            )
            prev_reward_layer_size1 = size
        self._reward_branch_separate1 = nn.Sequential(*reward__layers)

        self._reward_branch1 = SlimFC(
            in_size=prev_reward_layer_size1, out_size=1, initializer=normc_initializer(0.01), activation_fn=None
        )


    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self.device=obs.device

        self.obs = obs.reshape(obs.shape[0], -1)
        features = self._hidden_layers(self.obs)
        logits = self._logits(features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)

        return logits, state

    def logit(self,obs):
        obs = torch.Tensor(obs).to(self.device)

        self.obs = obs.reshape(obs.shape[0], -1)
        features = self._hidden_layers(self.obs)
        logits = self._logits(features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)

        return logits

    @override(TorchModelV2)
    def value_function(self,) -> TensorType:
        _value_out = torch.reshape(self._value_branch(self._value_branch_separate(self.obs)), [-1])

        return _value_out

    def reward(self,obs,action,next_obs=None,log_prob=None,dones=None,gamma=0.99):

        if type(obs) is torch.Tensor:
            obs_act=torch.concat([obs,action],dim=-1)
        else:
            obs_act=np.concatenate([obs,action],axis=-1)

            obs_act=torch.Tensor(obs_act).to(self.device)

        dis_result = torch.reshape(self._reward_branch(self._reward_branch_separate(obs_act)), [-1])

        if next_obs is not None:
            obs=torch.Tensor(obs).to(self.device)
            next_obs=torch.Tensor(next_obs).to(self.device)
            dones=torch.Tensor(dones).to(self.device)
            log_prob=torch.Tensor(log_prob).to(self.device)

            obs_next_obs=torch.concat([obs,next_obs],dim=0)

            h= torch.reshape(self._reward_branch1(self._reward_branch_separate1(obs_next_obs)), [-1])

            h_next=h[len(h)//2:]
            h_cur=h[:len(h)//2]

            f=dis_result+(1-dones)*(gamma * h_next - h_cur)

            exp_f=torch.exp(f)
            dis_result=exp_f/(exp_f+torch.exp(log_prob))

        else:
            dis_result=torch.sigmoid(dis_result)

        return dis_result

    #




