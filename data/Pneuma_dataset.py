from torch.utils.data import Dataset

import numpy as np

class PneumaDataset(Dataset):
    def __init__(self,cfg,type,meta_manager):

        model_cfg = cfg['model_params']

        self.step_time = model_cfg['step_time']

        self.history_num_frames= model_cfg["history_num_frames"]
        self.future_num_frames = model_cfg["future_num_frames"]

        self.interval=model_cfg['interval']
        self.sim_len=model_cfg['sim_len']
        self.goal_threshold_m=model_cfg["goal_threshold_m"]

        self.agent_feature_len = self.history_num_frames + 1

        self.meta_manager=meta_manager

        self.route_point_num=model_cfg['route_point_num']

        self.light_dim=3

        self.head_dim=self.route_point_num//3

        self.route_idx=self.history_num_frames+self.route_point_num

        self.light_idx=self.route_idx+self.light_dim

        self.agent_feature_len +=self.route_point_num+self.light_dim

        if type == "train":
            self.dates = range(1, 15)
        else:
            self.dates = range(15, 20)

        try:
            data = np.load('./data/meta/traj_data/' + type + "32_4.npz", allow_pickle=True)

            self.agent_array = data["agent_array"]

            self.index_array = data["index_array"]

        except:
            from .data_load import data_process

            self.agent_array,self.index_array=data_process(self.dates,self.meta_manager.edge_list)

            np.savez_compressed("./data/meta/traj_data/" + type + "32_4.npz",agent_array=self.agent_array,index_array=self.index_array)

        self.start_index = list(np.where(self.index_array[:, 1] == 0)[0])#0,20551

        self.start_index.append(len(self.index_array))

        self.ori_std = model_cfg['ori_std']

        if type == "train":
            self.data_noise_std=model_cfg["training_noise_std"]
        else:
            self.data_noise_std=0

    def __len__(self):
        return len(self.index_array)

    def __getitem__(self, id) -> dict:

        scene_dataset=self.get_scene_dataset(id,self.future_num_frames)

        history_data=self.get_hist(scene_dataset)

        fut_data=self.get_future(scene_dataset)

        data={**history_data,**fut_data}

        return data

    def get_scene_dataset(self,id,fut_len):

        all_index=self.index_array[id]

        date_index=all_index[0]

        max_index = self.start_index[date_index + 1] - (fut_len * self.interval + 1) - 1

        if id > max_index:
            id = max_index
            all_index=self.index_array[id]

        agent_array,agent_id_array,agent_index,type_array,goal,route=self.agent_array[date_index]

        time_index=all_index[1]

        agent_dataset=[]

        for t in range(time_index-(self.history_num_frames-1)*self.interval,time_index+fut_len*self.interval+1,self.interval):
            if t<0:
                pos=np.empty([0,2])
                ids=np.empty([0])
            else:
                pos = agent_array[agent_index[t]:agent_index[t + 1]]

                pos+=self.data_noise_std*np.random.randn(len(pos),2)

                ids = agent_id_array[agent_index[t]:agent_index[t + 1]]

            agent_dataset.append([pos,ids])

        return [agent_dataset,type_array,goal,route,id]

    def get_future(self,scene_dataset):

        agent_dataset,type_array,goal,route,id=scene_dataset

        agent_future=agent_dataset[self.history_num_frames:]

        cur_pos,cur_ids=agent_dataset[self.history_num_frames-1]

        target_positions=np.zeros([len(cur_ids),self.future_num_frames,3])

        for t in range(self.future_num_frames):

            all_pos,all_ids=agent_future[t]

            index=np.where(cur_ids == all_ids[:,None])

            index0=index[0]

            index1=index[1]

            target_positions[index1,t,:2]=all_pos[index0]

            target_positions[index1,t,2]=1

        all_time = self.index_array[id:id+self.interval*(self.future_num_frames+1):self.interval,2]

        real_time = (all_time - all_time[0])/1000.0

        cur_pos=np.concatenate([cur_pos,np.ones([len(cur_pos),1])],axis=-1)[:,None]

        target_positions=np.concatenate([cur_pos,target_positions],axis=1)

        pred_time=np.arange(1,1+self.future_num_frames)*self.step_time#*1000#+real_time[0]

        for i in range(len(target_positions)):
            mask=target_positions[i,:,2].astype(bool)
            if np.any(mask[1:]):
                real_time_masked=real_time[mask]
                target_positions_i_masked=target_positions[i][mask]
                target_positions[i,1:, 0] = np.interp(pred_time, real_time_masked, target_positions_i_masked[:, 0])
                target_positions[i,1:, 1] = np.interp(pred_time, real_time_masked, target_positions_i_masked[:, 1])

        target_positions=target_positions[:,1:]

        fut_data = {
            "positions": target_positions.astype(np.float32),
        }

        return fut_data

    def get_hist(self,scene_dataset,frame_index=0,is_training=True):

        agent_dataset,type_array,goal,route,id=scene_dataset

        agent_history=agent_dataset[frame_index:frame_index+self.history_num_frames]#start_0

        cur_pos,cur_ids=agent_history[-1]

        agent_polylines=np.zeros([len(cur_ids),self.agent_feature_len,3],dtype=np.float32)

        agent_polylines[:,0,:2]=cur_pos

        agent_polylines[:,0,2]=type_array[cur_ids]

        for t in range(self.history_num_frames-1):

            all_pos,all_ids=agent_history[t]

            index=np.where(cur_ids == all_ids[:,None])

            index0=index[0]

            index1=index[1]

            agent_polylines[index1,self.history_num_frames-1-t,:2]=all_pos[index0]

            agent_polylines[index1,self.history_num_frames-1-t,2]=1

        # if self.use_context_hist:
        #     agent_polylines[:,-self.history_num_frames-1:-1]=agent_polylines[:,:self.history_num_frames]

        # if is_training:
        #     perturb=np.random.randn(len(agent_polylines),1,2)*self.centroid_std#*decay#*np.arange(self.history_num_frames)[None,:,None]
        #
        #     cur_pos+=perturb[:, 0, :2]
        #
            #agent_polylines[:, :1, :2]+=perturb

            #cur_pos=agent_polylines[:, 0, :2]

        # agent_polylines[:, self.history_num_frames-1, :2]=cur_pos
        # agent_polylines[:, self.history_num_frames-1, 2]=1

        all_index = self.index_array[id + frame_index * self.interval]

        date_index = all_index[0]

        real_time = all_index[2]

        next_time = self.index_array[id + (frame_index + 1) * self.interval][2]

        next_rel_time = (next_time - real_time)/1000.0

        agent_num=len(cur_ids)

        route_point=route[cur_ids]

        route_width=route_point[:,:,2]

        #if is_training:
        #ori_pos=cur_pos+np.random.randn(len(cur_pos),2)#*self.random_ori
        #else:
        ori_pos=cur_pos+self.ori_std*np.random.randn(len(cur_pos),2)

        dist_to_ori=np.linalg.norm(ori_pos[:,None]-route_point[:,:,:2],axis=-1)-route_width

        # dist_to_proj=np.linalg.norm(nearest_route_points[:,None,:2]-route_point[:,:,:2],axis=-1)-route_width

        sort_index=np.argsort(dist_to_ori,axis=-1)[:,:self.route_point_num]

        #nearest_index=sort_index[:,0]#np.argmin(dist_to_proj,axis=-1)

        first_index=np.arange(agent_num)

        #nearest_route_points =route_point[first_index, nearest_index]

        nearest_n_route_points =route_point[first_index[:,None], sort_index]

        agent_polylines[:, self.history_num_frames:self.history_num_frames + self.route_point_num] = nearest_n_route_points[..., :3]

        # if self.use_heading:
        #     agent_polylines[:,self.history_num_frames+self.route_point_num:self.route_idx]=nearest_n_route_points[...,-1].reshape(-1,self.head_dim,3)

        nearest_route_points=nearest_n_route_points[:,0]

        nearest_edgeid=nearest_route_points[...,3].astype(int)

        nearest_edge_point=self.meta_manager.edges[nearest_edgeid]

        agent_polylines[:, self.route_idx:self.light_idx,:2]=nearest_edge_point

        light = self.meta_manager.traffic_light[self.dates[date_index]]

        mask = light[:, 0] != 0

        avail_light = light[mask]#/ 1000

        time_gap = avail_light[:, 1]

        rel_time=(real_time - avail_light[:, 0]) % time_gap

        light_array = -np.ones([light.shape[0]])

        light_array[mask] =rel_time

        light_array = np.concatenate([light_array[:, None], light[:, 1:]], axis=-1) / 1000

        light_state=light_array[nearest_edgeid]

        agent_polylines[:, self.route_idx:self.light_idx,2]=light_state

        #goal:

        goal_array=goal[cur_ids]

        goal_postions=goal_array[:,:2]

        #goal_types=goal_array[:,2].astype(bool)

        dist_to_goal = np.linalg.norm(cur_pos - goal_postions, axis=-1)

        reach_goals =  (dist_to_goal < self.goal_threshold_m)# & goal_types

        agent_polylines[:,-1,:2]=goal_postions

        if np.all(reach_goals):
            reach_goals[0]=False

        agent_polylines[:,-1,2]=reach_goals+1

        agent_polylines=agent_polylines.reshape(agent_polylines.shape[0], -1)

        agent_polylines=np.concatenate([agent_polylines,ori_pos],axis=-1)

        return {
            "agent_polylines": agent_polylines,
            "next_rel_time": next_rel_time
        }
