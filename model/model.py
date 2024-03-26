import torch
import torch.nn as nn
from .vae_model import VAE_Model

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.future_num_frames = cfg['future_num_frames']

        self.vae_model = VAE_Model(cfg)

        self.output_vel=cfg['output_vel']

        self.learner_w=cfg["learner_w"]

    def parameterize_gaussian(self,param):

        dist=self.vae_model.parameterize_gmm(param)

        return dist

    def forward(self,states,training=True):
        state=states[...,:-2]

        batch_ori_pos=states[...,-2:]

        agent_mask=state[...,-1].to(bool)

        masked_state=state[agent_mask]

        ori_pos=batch_ori_pos[agent_mask]

        goal_pos=masked_state[ :, -3:-1]

        rotate=self.vae_model.get_rotate(goal_pos,ori_pos)

        graph_state,node_pos=self.vae_model.get_graph_state(state,batch_ori_pos,ori_pos,rotate)

        if training and self.learner_w!=0:
            action_preds,state_info,ori_pos ,rotate =self.vae_model(graph_state,node_pos,ori_pos,rotate)
        else:
            action_preds, rel_relation = self.vae_model.policy_network(graph_state, node_pos, rotate)

            if self.learner_w!=0:
                state_info= (graph_state,rel_relation)
            else:
                state_info=(0,0)

        return action_preds.reshape(len(graph_state), self.future_num_frames, -1), ori_pos[:, None], rotate,state_info,agent_mask


    def get_action(self, state):

        action_preds,ori_pos,rotate,state_info,agent_mask=self.forward(state,training=False)

        if action_preds.shape[-1]==5:
            dist=self.parameterize_gaussian(action_preds)
            action_preds = dist.sample()

            if self.output_vel:
                action_preds=torch.cumsum(action_preds,dim=1)

            action_preds[:, :, :2] = torch.einsum("nta,nba->ntb", action_preds[:, :, :2], rotate)

            action_preds[:, :, :2] += ori_pos

        else:
            if rotate is not None:
                action_preds[:, :, :2] = torch.einsum("nta,nba->ntb", action_preds[:, :, :2], rotate)

            action_preds[:, :, :2] += ori_pos

        fut_positions=torch.zeros([state.shape[0],state.shape[1],action_preds.shape[-2],action_preds.shape[-1]],device=action_preds.device)

        fut_positions[agent_mask]=action_preds

        return {"fut_positions": fut_positions,"state_info":state_info}


