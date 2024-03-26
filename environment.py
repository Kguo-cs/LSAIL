import numpy as np
from ray.rllib.utils.framework import try_import_torch
import gym
from ray.rllib.env import MultiAgentEnv
import copy
import os
from evaluate.lqr_smoother import LQR

root_dir=os.getcwd()

torch, nn = try_import_torch()

class Env(MultiAgentEnv):
    def __init__(self, config={}):
        MultiAgentEnv.__init__(self)

        self.type = config["type"] if "type" in config else 'train'

        data = np.load( root_dir+"/data/meta/traj_data/" + self.type + "32_4.npz", allow_pickle=True)

        self.agent_array = data["agent_array"]

        self.index_array = data["index_array"]

        self.avail_time=len(self.index_array)

        self.sim_len=50
        self.interval=10
        self.hist_dim=config["hist_dim"] if "hist_dim" in config else 10
        self.route_dim=config["route_dim"] if "route_dim" in config else 50
        self.light_dim = 3
        self.goal_dim=1
        self.goal_d=config["goal_d"] if "goal_d" in config else 10
        self.dist_threshold=30

        self.nei_num=8

        self.nei_hist=3

        self.nei_dim=self.nei_num*self.nei_hist

        self.hist_idx=self.hist_dim+self.nei_dim

        self.route_idx=self.hist_idx+self.route_dim

        self.light_idx = self.route_idx + self.light_dim

        self.total_dim=self.light_idx+self.goal_dim

        obs_dim=self.total_dim*3-2

        self.fut_dim=config["fut_dim"] if "fut_dim" in config else 1

        self.act_dim=self.fut_dim*2

        self.hist=config['hist'] if "hist" in config else True

        self.observation_space = gym.spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(obs_dim,),
                dtype=np.float32
            )

        self.action_space = gym.spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(self.act_dim,),
                dtype=np.float32
            )

        if self.type == "train":
            self.dates = range(1, 15)
        else:
            self.dates = range(15, 20)

        self.offroad_w=-1

        self.ade_w=config["ade_w"] if "ade_w" in config else -0.01

        self.done_w=config["done_w"] if "done_w" in config else 0.01

        self.load_map(root_dir)

        self.std=config["std"] if "std" in config else 0

        self.step_time=0.4

        cfg={
            "future_num_frames": self.fut_dim,
            "acc_w" : 1,
            "control_w" : 0,
            "lqr_iter": 1,
            "max_linesearch_iter": 10,
            "step_time": self.step_time
        }

        self.LQR=LQR(cfg)

        self.proj=config["proj"] if "proj" in config else False

        self.smooth=config["smooth"] if "smooth" in config else False

        self.iter=0

        self.iter_n=config["iter_n"]  if "iter_n" in config else 1

        self.start_index = np.where(self.index_array[:, 1] == 0)[0]

    def load_map(self,root_dir):
        edge_list = list(np.load(root_dir+"/data/meta/networks/network_3m.npy", allow_pickle=True))

        self.edges = []

        for edge in edge_list:
            self.edges.append(np.concatenate([edge[:1], edge[-2:]], axis=0))

        self.edges = np.array(self.edges)[...,:2]

        self.traffic_light = np.load(root_dir+"/data/meta/traffic_light/traffic_light.npy")

        all_point = np.load(root_dir+"/data/meta/networks/point_3m.npy")

        self.all_point=all_point[:, :2]

        self.all_point_width=all_point[:, 2] + 1.5

    def get_scene_dataset(self,id,fut_len):

        all_index=self.index_array[id]

        date_index=all_index[0]

        agent_array,agent_id_array,agent_index,type_array,goal,route=self.agent_array[date_index]

        time_index=all_index[1]

        agent_dataset=[]

        max_index=len(agent_index)-(fut_len*self.interval+1)-1#len(agent_index)

        time_index=min(time_index,max_index)

        for t in range(time_index-(self.hist_dim-1)*self.interval,time_index+fut_len*self.interval+1,self.interval):
            if t<0:
                pos=np.empty([0,2])
                ids=np.empty([0])
            else:
                pos = agent_array[agent_index[t]:agent_index[t + 1]]
                ids = agent_id_array[agent_index[t]:agent_index[t + 1]]

            agent_dataset.append([pos,ids])

        return [agent_dataset,type_array,goal,route,id]

    def get_nei_obs(self,cur_pos,hist_pos):

        src_dst_dist=np.linalg.norm(cur_pos[:,None]-cur_pos[None],axis=-1)

        nearest_indices = np.argsort(src_dst_dist,axis=-1)[:,1:self.nei_num+1]

        nearest_agent=hist_pos[nearest_indices,:,:]

        nei_obs=nearest_agent.reshape(len(nearest_agent),-1,3)

        return nei_obs

    def get_obs(self,frame_index):

        agent_history=self.agent_dataset[frame_index:frame_index+self.hist_dim]#start_0

        cur_pos,cur_ids=agent_history[-1]

        observations=np.zeros([len(cur_ids),self.total_dim,3],dtype=np.float32)

        observations[:,0,:2]=cur_pos

        observations[:,0,2]=self.type_array[cur_ids]

        for t in range(1,self.hist_dim):

            all_pos,all_ids=agent_history[self.hist_dim-1-t]

            index=np.where(cur_ids == all_ids[:,None])

            index0=index[0]

            index1=index[1]

            observations[index1,t,:2]=all_pos[index0]

            observations[index1,t,2]=1

        observations[:,:self.hist_dim,:2]+=np.random.randn(len(observations),self.hist_dim,2)*self.std

        cur_pos=observations[:, 0, :2]

        nei_obs=self.get_nei_obs(cur_pos,observations[:,:self.nei_hist])

        observations[:,self.hist_dim:self.hist_dim+nei_obs.shape[1]]=nei_obs

        agent_num=len(cur_ids)

        route_point=self.route[cur_ids]

        dist_to_proj=np.linalg.norm(cur_pos[:,None]-route_point[:,:,:2],axis=-1)

        route_width=route_point[:,:,2]

        dist_to_proj=dist_to_proj-route_width

        sort_index=np.argsort(dist_to_proj,axis=-1)[:,:self.route_dim]

        first_index=np.arange(agent_num)[:,None]

        nearest_route_points =route_point[first_index, sort_index]

        observations[:, self.hist_idx:self.route_idx] = nearest_route_points[...,:3]

        nearest_point_edge=nearest_route_points[...,0,3].astype(int)

        nearest_edge_point=self.edges[nearest_point_edge]

        observations[:, self.route_idx:self.light_idx,:2]=nearest_edge_point[:,:self.light_dim,:2]

        all_index = self.index_array[min(self.id + frame_index * self.interval, len(self.index_array) - 1)]

        date_index = all_index[0]

        real_time = all_index[2]

        #next_time = self.index_array[min(id + (frame_index + 1) * self.interval, len(self.index_array) - 1)][2]

        #next_real_time = next_time.astype(int) - real_time.astype(int)

        light = self.traffic_light[self.dates[date_index]]

        mask = light[:, 0] != 0

        avail_light = light[mask]#/ 1000

        time_gap = avail_light[:, 1]

        rel_time=(real_time - avail_light[:, 0]) % time_gap

        light_array = -np.ones([light.shape[0]])

        light_array[mask] =rel_time

        light_array = np.concatenate([light_array[:, None], light[:, 1:]], axis=-1) / 1000

        light_state=light_array[nearest_point_edge]

        observations[:, self.route_idx:self.light_idx,2]=light_state

        goal_array=self.goal[cur_ids]

        goal_pos=goal_array[:,:2]

        observations[:,-1,:2]=goal_pos

        observations[:,-1,2]=1

        rel_pos = goal_pos - cur_pos

        agent_yaw_rad = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])

        cos = np.cos(agent_yaw_rad)
        sin = np.sin(agent_yaw_rad)

        rotate1 = np.stack([cos, -sin], axis=1)
        rotate2 = np.stack([sin, cos], axis=1)

        rotate = np.stack([rotate1, rotate2], axis=1).astype(np.float32)

        rel_state = observations[:, 1:, :2] - cur_pos[:, None, :2]

        if not self.hist:
            observations[:, 1:self.hist_dim,2] = 0
            observations[:, self.hist_dim+1:self.hist_idx:3,2] = 0
            observations[:, self.hist_dim+2:self.hist_idx:3,2] = 0

        state_mask = observations[:, 1:self.route_idx, 2:3]

        rel_state = np.einsum("nta,nab->ntb", rel_state, rotate)

        rel_state[:, :state_mask.shape[1], :2] *= state_mask.astype(bool)

        state_attr = observations[..., 2]

        state = np.concatenate([rel_state.reshape(agent_num, -1), state_attr], axis=-1)

        obs={}

        for i,track_id in enumerate(cur_ids):
            obs[track_id]=state[i]

        self.cur_ids=cur_ids

        self.cur_pos=cur_pos

        self.rotate=rotate

        self.goal_pos=goal_pos

        self.route_width=nearest_route_points[:,:,2]

        self.route_point=nearest_route_points[:,:,:2]

        self.hist_traj=observations[:,:2,:2]

        return obs

    def reset(self):
        #print(self.type,"reset")

        self.frame_index=0

        if self.iter%self.iter_n==0:
            self.id = np.random.randint(low=0,high=self.avail_time-1)

        self.iter+=1

        agent_dataset,self.type_array,self.goal,self.route,self.id = self.get_scene_dataset(self.id, self.sim_len)

        self.record_dataset =  agent_dataset[self.hist_dim:]

        self.agent_dataset=copy.deepcopy(agent_dataset)

        obs=self.get_obs(self.frame_index)

        self._agent_ids=obs.keys()

        return obs

    def reset_id(self,id):
        self.id=id
        self.frame_index=0

        agent_dataset,self.type_array,self.goal,self.route,self.id = self.get_scene_dataset(self.id, self.sim_len)

        self.record_dataset =  agent_dataset[self.hist_dim:]

        self.agent_dataset=copy.deepcopy(agent_dataset)

        obs=self.get_obs(self.frame_index)

        self._agent_ids=obs.keys()

        return obs

    def get_reward(self,next_pos):

        distance = np.linalg.norm(next_pos[:, None] - self.all_point[None], axis=-1)

        distance = distance - self.all_point_width[None]

        min_distance = np.amin(distance, axis=1)

        off_road = (min_distance > 0).astype(np.float32)

        target_pos=copy.deepcopy(self.goal_pos)

        real_pos, self.real_ids = self.record_dataset[self.frame_index]

        index = np.where(self.cur_ids == self.real_ids[:, None])

        index0 = index[0]

        index1 = index[1]

        target_pos[index1] = real_pos[index0]

        ade = np.linalg.norm(next_pos - target_pos, axis=-1)

        return off_road,ade,target_pos

    def update_dataset(self,next_pos,frame_idx):

        dist_to_goal = np.linalg.norm(next_pos - self.goal_pos, axis=-1)  # not reach goal and avail

        not_reach_goal = dist_to_goal > self.goal_d

        real_next_pos,real_next_ids=self.agent_dataset[frame_idx]

        new_ids=set(real_next_ids)-self._agent_ids#new agent

        self._agent_ids|=  set(real_next_ids)#all finished_agent

        next_position=next_pos[not_reach_goal]

        if np.all(not_reach_goal==False):
            not_reach_goal[0]=True

        next_ids=self.cur_ids[not_reach_goal]

        if len(new_ids):
            new_ids=np.array(list(new_ids))

            index = np.where(real_next_ids == new_ids[:, None])[1]

            new_pos = real_next_pos[index]

            self.agent_dataset[frame_idx][0] =np.concatenate([next_position,new_pos],axis=0)
            self.agent_dataset[frame_idx][1] =np.concatenate([next_ids,new_ids],axis=0)
        else:
            self.agent_dataset[frame_idx][0] =next_position
            self.agent_dataset[frame_idx][1] =next_ids

        reach_goals=~not_reach_goal

        done=dict(zip(self.cur_ids,reach_goals.T))

        ternimal=reach_goals.astype(np.float32)

        off_road,ade,target_pos=self.get_reward(next_pos)

        rew=self.done_w*ternimal+self.ade_w*ade+self.offroad_w*off_road

        # rel_futpos = target_pos - self.cur_pos
        #
        # fut_postions = np.einsum("na,nab->nb", rel_futpos, self.rotate).astype(np.float32)
        #
        # rew=np.round(fut_postions[:,0]* 10e1)+fut_postions[:,1]/10e2

        # first_pos = np.round(rew) / 10e1
        #
        # second_pos = (rew - np.round(rew)) * 10e2

        # x=fut_postions[:,0]-first_pos
        #
        # y=fut_postions[:,1]-second_pos

        info={}

        for i,id in enumerate(self.cur_ids):
            info[id]={"offroad":off_road[i],"ade":ade[i],"episode_reward":rew[i]}

        rew=dict(zip(self.cur_ids,rew.T))

        return done,rew,info

    def step(self, action_dict):

        actions= np.stack(list(action_dict.values()),axis=0)#np.array(action_dict.values())

        action_preds=actions.reshape(len(actions),-1,2)

        action_preds = np.einsum("nta,nba->ntb", action_preds, self.rotate)

        action_preds += self.cur_pos[:,None]

        action_preds=self.post_process(action_preds)

        next_pos=action_preds[:,0]

        done,rew,info=self.update_dataset(next_pos,self.frame_index+self.hist_dim)

        self.frame_index+=1

        obs=self.get_obs(self.frame_index)

        info={key: value for key, value in info.items() if key in self.cur_ids}

        done["__all__"] = self.frame_index == self.sim_len

        # if self.gail:
        #     obs,actions,dones=self.offline_data.sample()
        #
        #     print(1)
        return obs, rew, done, info

    def sample(self):
        id = np.random.randint(low=0,high=self.avail_time-1)

        self.agent_dataset,self.type_array,self.goal,self.route,self.id = self.get_scene_dataset(id, self.fut_dim)

        observations=self.get_obs(frame_index=0)

        agent_future=self.agent_dataset[self.hist_dim:]

        fut_postions=np.zeros([len(self.cur_ids),self.fut_dim,2],dtype=np.float32)
        fut_avails=np.zeros([len(self.cur_ids),self.fut_dim])

        for t in range(self.fut_dim):

            all_pos,all_ids=agent_future[t]

            index=np.where(self.cur_ids == all_ids[:,None])

            index0=index[0]

            index1=index[1]

            fut_postions[index1,t,:2]=all_pos[index0]

            fut_avails[index1,t]=1

        rel_futpos = fut_postions - self.cur_pos[:,None]

        fut_postions = np.einsum("nta,nab->ntb", rel_futpos, self.rotate)

        all_avial=np.all(fut_avails[:,:],axis=1)

        obs=np.stack(list(observations.values()),axis=0)

        obs=obs[all_avial]

        actions=fut_postions[all_avial].reshape(len(obs),-1)

        dones=np.ones_like(actions[:,0]).astype(bool)

        return obs,actions,dones

    def post_process(self, action_preds):

        action_preds=torch.FloatTensor(action_preds)

        if self.proj:
            action_preds=self.road_proj(action_preds)

        if self.smooth:
            hist_traj = torch.FloatTensor(self.hist_traj)

            speed_mask = hist_traj[:, 1, -1].to(bool)

            hist_traj = hist_traj[speed_mask]

            cur_pos = hist_traj[:, 0, :2]

            prev_pos = hist_traj[:, 1, :2]

            prev_vel = (cur_pos - prev_pos) / self.step_time

            x_init = torch.cat([cur_pos, prev_vel], dim=-1)

            if len(x_init):

                action_preds[speed_mask] = self.LQR(x_init, action_preds[speed_mask])

        return action_preds.numpy()

    def road_proj(self, point):

        route_point = torch.FloatTensor(self.route_point)

        route_width = torch.FloatTensor(self.route_width)

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

        proj_point = line_start + torch.einsum("nab,nab->na", point - line_start, line_dir_norm)[:, :,
                                        None] * line_dir_norm

        proj_to_point = point - proj_point

        nearest_distance = torch.linalg.norm(proj_to_point, dim=-1)

        new_action_preds = torch.clamp_max(width / nearest_distance, max=1)[:, :, None] * proj_to_point + proj_point

        return new_action_preds

    def action_space_sample(self,agent_ids):

        action_dict={}

        for id in agent_ids:
            action_dict[id]=np.zeros([self.act_dim])

        return action_dict





# env=Env({"type":"val"})
# #
# # # obs=env.reset()
# #
# #
# while True:
#     env.sample(78280)
# #81314 train,78280 val


    # action_dict={}
    #
    # for i in obs.keys():
    #     action_dict[i]=np.zeros([2])
    #
    # obs=env.step(action_dict)[0]
