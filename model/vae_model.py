from torch import nn
import torch
from .graphNet import GraphNet,rep2batch

class VAE_Model(nn.Module):

    def __init__(self,cfg):
        super(VAE_Model, self).__init__()

        self.history_num_frames=cfg['history_num_frames']

        self.route_point_num=cfg["route_point_num"]

        agent_feat_dim =3 * (self.history_num_frames + 4 + cfg["route_point_num"])

        future_num_frames=cfg["future_num_frames"]
        output_gaussian=cfg['output_gaussian']

        if output_gaussian:
            act_dim=5
            self.param_dim = 2

            self.tril_size = (self.param_dim + 1) * self.param_dim // 2
            self.actout_dim = self.param_dim+self.tril_size+1

            self.tril_mask = torch.zeros([self.param_dim,self.param_dim])
            self.tril_mask[tuple(torch.tril_indices(self.param_dim, self.param_dim,offset=-1))] = 1

            self.diag_mask = torch.eye(self.param_dim).to(bool)
        else:
            act_dim=2

        self.policy_network = GraphNet(cfg, agent_feat_dim, act_dim * future_num_frames,
                                       layer_num=cfg["global_num_layers"])  # MLP(agent_feat_dim,d_model,d_model) #

        self.route_dim = self.history_num_frames + self.route_point_num

        self.learner_w=cfg["learner_w"]

        self.ori_std = cfg['ori_std']

        if self.learner_w!=0:
            latent_dim = cfg['latent_dim']
            vae_layer_num=cfg["vae_layer_num"]

            self.encoder = GraphNet(cfg, agent_feat_dim, 2 * latent_dim,layer_num=vae_layer_num)
            self.decoder = GraphNet(cfg, latent_dim + agent_feat_dim - (self.history_num_frames * 2),
                                    self.history_num_frames * 5,layer_num=vae_layer_num)

    def get_graph_state(self,batch_state,batch_ori_pos,ori_pos,rotate):

        agent_mask=batch_state[...,-1].to(bool)

        node_pos = torch.zeros_like(batch_state[:, :, :2])

        for i in range(len(node_pos)):
            node_pos[i] = 100000 * i + batch_ori_pos[i]

        state = batch_state[agent_mask]

        agent_num=len(state)

        state_reshape = state.view(agent_num, -1, 3)

        rel_pos=state_reshape[:, :, :2]

        state_attr =state_reshape[:, :, 2]

        rel_state = rel_pos - ori_pos[:, None]

        rel_state = torch.einsum("nta,nab->ntb", rel_state, rotate)

        state_mask = state_attr[:, :self.route_dim,None].to(bool)

        # if self.use_hist_vel:
        #     rel_hist = rel_state[:, :self.history_num_frames]
        #
        #     vel=rel_hist[:,1:]-rel_hist[:,:-1]
        #
        #     hist_mask=state_mask[:,:self.history_num_frames]
        #
        #     vel_mask=hist_mask[:,1:]&hist_mask[:,:-1]
        #
        #     vel=vel*vel_mask
        #
        #     rel_state[:, 1:self.history_num_frames]=vel

        rel_state[:, :state_mask.shape[1]] *= state_mask

        state = torch.cat([rel_state, state_attr[:,:,None]], dim=-1)

        return state.view(agent_num, -1),node_pos[agent_mask]

    def get_rotate(self,goal_pos,ori_pos):

        rel_pos = goal_pos - ori_pos

        agent_yaw_rad = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])

        cos = torch.cos(agent_yaw_rad)
        sin = torch.sin(agent_yaw_rad)

        rotate1 = torch.stack([cos, -sin], dim=1)
        rotate2 = torch.stack([sin, cos], dim=1)

        rotate = torch.stack([rotate1, rotate2], dim=1)

        return rotate

    def parameterize_gmm(self, param):

        mean = param[..., :2]

        batch_shape = param.shape[:-1]

        tril_mask = rep2batch(self.tril_mask, batch_shape).to(param.device)

        tril = torch.zeros_like(tril_mask)

        tril[tril_mask.to(bool)] = param[..., self.param_dim:self.tril_size].flatten()

        diag_mask = rep2batch(self.diag_mask, batch_shape).to(param.device)

        std_param =param[..., self.tril_size:self.param_dim + self.tril_size].flatten()

        # if clip==True:,eps=0.0
        #     std_param=torch.clamp_max(std_param, max=self.std_clip)

        tril[diag_mask] = torch.exp(std_param) #+ eps

        dist = torch.distributions.MultivariateNormal(loc=mean, scale_tril=tril)

        return dist

    def get_rec_state(self,expert_rec_hist_dist,state_attr,hist_mask,context_state,node_pos,ori_pos,rotate):

        agent_num = len(state_attr)

        expert_rec_hist = (expert_rec_hist_dist.sample()* hist_mask[:, :, None]).detach()

        # if self.use_hist_vel:
        #
        #     expert_rec_hist=torch.cumsum(expert_rec_hist,dim=1)

        expert_rec_rel_state = torch.cat([expert_rec_hist, context_state], dim=1)

        expert_rec_state_attr=state_attr.clone()

        noise=self.ori_std*torch.randn_like(expert_rec_rel_state[:, 0])

        expert_rec_cur = expert_rec_rel_state[:, 0]+noise

        route_point_pos = context_state[:, :self.route_point_num]

        route_width = state_attr[:, self.history_num_frames:self.route_dim]

        route_point_avail = route_width.to(bool)

        routing_dist = torch.linalg.norm(route_point_pos - expert_rec_cur[:, None],  dim=-1) - route_width - route_point_avail * 1000

        sort_index = torch.argsort(routing_dist, dim=-1)

        first_index = torch.arange(agent_num)[:, None]

        expert_rec_rel_state[:, self.history_num_frames:self.route_dim] = route_point_pos[first_index, sort_index]

        expert_rec_state_attr[:, self.history_num_frames:self.route_dim] = route_width[first_index, sort_index]

        expert_rec_abs_state = torch.einsum("nta,nba->ntb", expert_rec_rel_state, rotate) + ori_pos[:, None]

        noise = torch.einsum("na,nba->nb", noise, rotate)

        expert_abs_goal = expert_rec_abs_state[:, -1]

        expert_rec_ori_pos = expert_rec_abs_state[:, 0]+noise

        expert_rec_rotate = self.get_rotate(expert_abs_goal, expert_rec_ori_pos)

        expert_rec_rel_state = torch.einsum("nta,nab->ntb", expert_rec_abs_state - expert_rec_ori_pos[:, None], expert_rec_rotate)

        expert_rec_state_mask = expert_rec_state_attr[:, :self.route_dim, None].to(bool)

        expert_rec_rel_state[:, :self.route_dim] *= expert_rec_state_mask

        expert_rec_node_pos = node_pos + expert_rec_ori_pos - ori_pos

        expert_rec_graph_state = torch.cat([expert_rec_rel_state, state_attr[:, :, None]], dim=-1)

        return expert_rec_graph_state.view(agent_num, -1),expert_rec_node_pos,expert_rec_ori_pos, expert_rec_rotate

    def forward(self,graph_state,node_pos,ori_pos,rotate):

        vae_loss,expert_rec_hist_dist,state_attr,hist_mask,context_state = self.vae_rec_hist(graph_state, node_pos,rotate)

        expert_rec_graph_state,expert_rec_node_pos,expert_rec_ori_pos, expert_rec_rotate=self.get_rec_state(expert_rec_hist_dist,state_attr,hist_mask,context_state,node_pos,ori_pos,rotate)

        action_preds, rel_relation = self.policy_network(expert_rec_graph_state, expert_rec_node_pos, expert_rec_rotate)

        return action_preds, vae_loss, expert_rec_ori_pos, expert_rec_rotate

    def vae_rec_hist(self,graph_state, node_pos=None,   rotate=None,rel_relation=None):

        agent_num=len(graph_state)

        state_reshape=graph_state.view(agent_num,-1,3)

        ori_rel_state=state_reshape[:, :,:2]

        state_attr=state_reshape[:, :,2]

        ori_hist = ori_rel_state[:, :self.history_num_frames]

        context_state= ori_rel_state[:, self.history_num_frames:]

        h,rel_relation = self.encoder(graph_state, node_pos,rotate,rel_relation)

        mu, logvar = h.chunk(2, dim=1)

        epsilon = torch.randn_like(mu)

        z = mu + epsilon * torch.exp(logvar / 2)

        z = torch.cat([z, context_state.reshape(agent_num,-1), state_attr.view(agent_num, -1)], dim=-1)

        rec_hist_dist_param =self.decoder(z,rel_relation=rel_relation)[0].view(agent_num, self.history_num_frames , -1)

        rec_hist_dist = self.parameterize_gmm(rec_hist_dist_param)

        kl_loss=0.5 * torch.sum(logvar.exp()+ mu.pow(2)- logvar-1,dim=-1).mean()

        neg_logp = -rec_hist_dist.log_prob(ori_hist)

        hist_mask = state_attr[:, :self.history_num_frames].to(bool)

        # if self.use_hist_vel:
        #     hist_mask[:,1:]&=hist_mask[:,:-1]

        rec_sum=neg_logp[hist_mask]#.mean() #torch.sum(logp*hist_mask,dim=-1)#

        # rec_sum=torch.clamp_max(rec_sum,self.max_clip)

        rec_loss=rec_sum.mean()

        # vae_loss=rec_loss+kl_loss

        return (rec_loss,kl_loss),rec_hist_dist,state_attr,hist_mask,context_state

