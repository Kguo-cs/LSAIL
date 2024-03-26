import numpy as np
from tqdm.auto import tqdm
import torch
from environment import Env
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian


def move_to_device(data, device: torch.device):
    return {k: v.to(device) for k, v in data.items()}


class ClosedLoopSimulator:
    def __init__(self,
                 dataset,
                 model,
                 cfg,
                 buffer=None,
                 evaluator = None
            ):


        self.env = Env(config={"type": "vala", "iter_n": 1, "proj": cfg["proj_on_road"], "smooth": cfg["smooth"],
                               "fut_dim": cfg["future_num_frames"]})

        self.model=model

        self.sim_len=dataset.sim_len

        self.evaluator=evaluator

        self.history_num_frames=dataset.history_num_frames



    def longterm_unroll(self,device,i):

        scene_idx=self.env.start_index[i]

        self.env.sim_len=2000

        sim_outputs = self.unroll([scene_idx],device, sim_len=2000,longterm=True)

        return sim_outputs

    def unroll(self, scene_indices,device,sim_len=None,training=False,longterm=False,rollout_time=1):

        sim_out_list=[]
        if sim_len == None:
            sim_len = self.sim_len

        for _ in range(rollout_time):
            sim_outputs=[]

            for scene in scene_indices:

                state=self.env.reset_id(scene)

                engaged_ids=set(state.keys())

                edge_para_list = []
                #off_road_list=[]

                for frame_idx in tqdm(range(sim_len), disable=False):
                    obs=np.stack(list(state.values()),axis=0)
                    obs_flat=torch.FloatTensor(obs).to(device)

                    if longterm:
                        rel_state=obs_flat[:,:6].reshape(-1,3,2)

                        cur_pos = torch.FloatTensor(self.env.cur_pos).to(device)

                        rotate = torch.FloatTensor(self.env.rotate).to(device)

                        action_preds = torch.einsum("nta,nba->ntb", rel_state, rotate)

                        action_preds += cur_pos[:, None]

                        obs_len=obs_flat.shape[1] // 3*2

                        agent_mask=obs_flat[:,obs_len:obs_len+3]

                        abs_obs=torch.cat([action_preds,agent_mask[:,:,None]],dim=-1).reshape(-1,9)

                        edge_parameter= self.evaluator.get_edge_res(abs_obs)

                        edge_para_list.append(edge_parameter)

                        # off_road_list.append(off_road)

                    state_dict={"obs":obs_flat}

                    logits = self.model(state_dict)[0]

                    # class 'ray.rllib.models.torch.torch_action_dist.TorchDiagGaussian'>, action_space=Box(-inf, inf, (2, ), float32))
                    curr_action_dist = TorchDiagGaussian(logits, self.model)

                    action=curr_action_dist.sample()

                    action_dict = dict(zip(state.keys(), action.cpu().numpy()))

                    engaged_ids|=set(state.keys())

                    state, rew, done, info=self.env.step(action_dict)


                id= self.env.id
                engaged_ids = np.array(list(engaged_ids))

                all_index = self.env.index_array[id]
                date_index = self.env.dates[all_index[0]]
                real_time = all_index[2]

                sim_out=(self.env.agent_dataset[self.history_num_frames:],self.env.record_dataset,self.env.type_array,engaged_ids, date_index,    real_time)

                if longterm:
                    return edge_para_list, sim_out
                else:
                    sim_outputs.append( sim_out)

            sim_out_list.append(sim_outputs)

        return sim_out_list
