import copy

import numpy as np
from tqdm.auto import tqdm
import torch
from data.feature_collate import FeatureCollate
from .lqr_smoother import LQR
#from static_vis import vis

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

            self.model = model

            self.dataset = dataset

            self.sim_len = dataset.sim_len

            self.history_num_frames = dataset.history_num_frames

            self.collate = FeatureCollate()

            self.buffer = buffer

            self.LQR = LQR(cfg)

            self.step_time = dataset.step_time

            self.smooth = cfg["smooth"]

            self.control = cfg["control_w"] != 0

            self.proj_on_road = cfg["proj_on_road"]

            self.evaluator = evaluator

            self.route_point_num = cfg['route_point_num']

    def longterm_unroll(self,device,i):

        scene_idx=self.dataset.start_index[i]

        sim_outputs = self.unroll([scene_idx],device, sim_len=2000,longterm=True)

        return sim_outputs

    def unroll(self, scene_indices,device,sim_len=None,training=False,longterm=False,rollout_time=1):

        sim_out_list=[]
        if sim_len == None:
            sim_len = self.sim_len

        for _ in range(rollout_time):

            scene_datasets=[]

            real_datasets=[]

            engaged_ids_list=[]

            for scene_idx in scene_indices:

                scene_dataset=self.dataset.get_scene_dataset(scene_idx,sim_len)

                sim_dataset=copy.deepcopy(scene_dataset)

                scene_datasets.append(sim_dataset)

                recorded=scene_dataset[0][self.history_num_frames:]#not include current step

                real_datasets.append(recorded)

                engaged_ids_list.append(set(scene_dataset[0][self.history_num_frames-1][1]))#current step is self.history_num_frames-1

            edge_res_list = []

            for frame_idx in tqdm(range(sim_len), disable=True):
                ego_input = []

                for scene_dataset in scene_datasets:

                    inputs=self.dataset.get_hist(scene_dataset,frame_idx,False)

                    ego_input.append(inputs)

                ego_input_dict = self.collate(ego_input)

                input_dict = move_to_device(ego_input_dict, device)

                state=input_dict["agent_polylines"]

                if longterm:
                    edge_res= self.evaluator.get_edge_res(state[0])

                    edge_res_list.append(edge_res)

                output_dict = self.model.get_action(state)

                self.update_agents(scene_datasets,engaged_ids_list, frame_idx+self.history_num_frames, input_dict, output_dict)

                if training:
                    self.buffer.append(output_dict['state_info'])

            if not training:
                sim_outputs=[]
                for i in range(len(scene_datasets)):
                    agent_dataset,type_array,goal,route,id= scene_datasets[i]
                    engaged_ids=np.array(list(engaged_ids_list[i]))
                    all_index=self.dataset.index_array[id]
                    date_index = self.dataset.dates[all_index[0]]
                    real_time = all_index[2]

                    sim_out=(agent_dataset[self.history_num_frames:], real_datasets[i],type_array, engaged_ids,date_index,real_time)

                    if longterm:
                        return  edge_res_list,sim_out
                    else:
                        sim_outputs.append(sim_out)

                sim_out_list.append(sim_outputs)

        return sim_out_list

    def update_agents(self,scene_datasets,engaged_ids_list, frame_idx: int, input_dict, output_dict):

        state=input_dict["agent_polylines"][...,:-2]

        fut_positions=output_dict["fut_positions"]

        fut_planned=self.post_process(state,fut_positions)

        # print(torch.max(fut_planned))

        # rel_pos=state.reshape(state.shape[0],state.shape[1],-1,3)[0,:,:,:2]
        #
        # state_attr=state.reshape(state.shape[0],state.shape[1],-1,3)[0,:,:,2]
        #
        # vis(rel_pos.cpu().numpy(), state_attr.cpu().numpy().astype(bool),fut_positions[0].cpu().numpy(),fut_planned[0].cpu().numpy())


        next_rel_times=input_dict["next_rel_time"].cpu().numpy()/0.4

        pred_positions=fut_planned[:,:, :,:2].cpu().numpy()

        cur_pos=state[...,None,:2].cpu().numpy()

        pred_positions=np.concatenate([cur_pos,pred_positions],axis=2)

        reach_goals=state[:,:,-1].cpu().numpy()

        for (scene_dataset,next_rel_time, pred_position,reach_goal,engaged_ids) in zip(scene_datasets,next_rel_times, pred_positions,reach_goals,engaged_ids_list):

            if next_rel_time>1 and pred_position.shape[1] > 2:
                pred_next_position=pred_position[:,0]+(next_rel_time)*(pred_position[:,1]-pred_position[:,0])
            else:
                pred_next_position =next_rel_time*pred_position[:,1]+(1-next_rel_time)*pred_position[:,0]

            not_reach_goal = reach_goal == 1  # not reach goal and avail

            real_next_pos,real_next_ids=scene_dataset[0][frame_idx]

            cur_ids=scene_dataset[0][frame_idx-1][1]

            new_ids=set(real_next_ids)-engaged_ids#new agent

            engaged_ids|=  set(real_next_ids)#all finished_agent

            next_position=pred_next_position[not_reach_goal]

            next_ids=cur_ids[not_reach_goal[ reach_goal.astype(bool)]]

            if len(new_ids):
                new_ids=np.array(list(new_ids))

                index = np.where(real_next_ids == new_ids[:, None])[1]

                new_pos = real_next_pos[index]

                # mask=~((new_pos[:,0]>1080) & (new_pos[:,0]< 1200) & (new_pos[:,1]>1500) & (new_pos[:,1]<1570))
                # mask=~((new_pos[:,0]>1025) & (new_pos[:,0]< 1215) & (new_pos[:,1]>1410) & (new_pos[:,1]<1485))
                #
                # new_pos=new_pos[mask]
                # new_ids=new_ids[mask]

                scene_dataset[0][frame_idx][0] =np.concatenate([next_position,new_pos],axis=0)
                scene_dataset[0][frame_idx][1] =np.concatenate([next_ids,new_ids],axis=0)
            else:
                scene_dataset[0][frame_idx][0] =next_position
                scene_dataset[0][frame_idx][1] =next_ids

    def post_process(self,state,fut_positions):

        if self.proj_on_road or self.smooth:
            mask = state[..., -1].to(bool)

            new_action_preds = fut_positions[mask]

            state_masked = state[mask].view(len(new_action_preds), -1, 3)

            if self.proj_on_road:
                new_action_preds=self.road_proj(new_action_preds,state_masked)

            if self.smooth:

                new_action_preds=self.smooth_traj(state_masked,new_action_preds)

            fut_positions = torch.zeros([state.shape[0], state.shape[1], fut_positions.shape[-2], 2],
                                          device=fut_positions.device)

            fut_positions[mask] = new_action_preds

        return fut_positions

    def smooth_traj(self,hist_traj,new_action_preds):

        speed_mask = hist_traj[:, 1, -1].to(bool)

        hist_traj = hist_traj[speed_mask]

        cur_pos = hist_traj[:, 0, :2]

        prev_pos = hist_traj[:, 1, :2]

        prev_vel = (cur_pos - prev_pos) / self.step_time

        x_init = torch.cat([cur_pos, prev_vel], dim=-1)

        if len(x_init):
            new_action_preds[speed_mask] = self.LQR(x_init, new_action_preds[speed_mask])

        return new_action_preds

    def road_proj(self, point, state_masked):

        route = state_masked[:, self.history_num_frames:self.history_num_frames+self.route_point_num]

        route_point = route[:, :, :2]

        route_width = route[:, :, 2]

        route_distance = torch.linalg.norm(point[:, :, None] - route_point[:, None], dim=-1) -route_width[:,None]

        nearest_point= torch.argsort(route_distance, dim=-1)

        nearest_point1 = nearest_point[:, :, 0]
        nearest_point2 = nearest_point[:, :, 1]

        first_index = torch.arange(len(point))[:, None]

        line_start = route_point[first_index, nearest_point1]

        line_end = route_point[first_index, nearest_point2]

        route_width1 = route_width[first_index, nearest_point1]

        route_width2 = route_width[first_index, nearest_point2]

        width =(route_width1+route_width2)/2

        line_dir = line_end - line_start

        line_len= torch.linalg.norm(line_dir, dim=-1)

        line_len=torch.clamp_min(line_len,0.001)

        line_dir_norm = line_dir /line_len[:,:,None]

        proj_point = line_start + torch.einsum("nab,nab->na", point - line_start, line_dir_norm)[:, :, None] * line_dir_norm

        proj_to_point = point - proj_point

        nearest_distance = torch.linalg.norm(proj_to_point, dim=-1)

        new_action_preds = torch.clamp_max(width / nearest_distance, max=1)[:, :, None] * proj_to_point + proj_point

        return new_action_preds