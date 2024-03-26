import torch
import numpy as np
from torch_scatter import scatter_add
from multiprocessing import Pool
import os
from shapely.geometry import  Polygon


def _get_bounding_box(centroid: np.ndarray, yaw: np.ndarray,
                      extent: np.ndarray,) -> Polygon:
    """This function will get a shapely Polygon representing the bounding box
    with an optional buffer around it.

    :param centroid: centroid of the agent
    :param yaw: the yaw of the agent
    :param extent: the extent of the agent
    :return: a shapely Polygon
    """
    x, y = centroid[0], centroid[1]
    sin, cos = np.sin(yaw), np.cos(yaw)
    width, length = extent[0] / 2, extent[1] / 2

    x1, y1 = (x + width * cos - length * sin, y + width * sin + length * cos)
    x2, y2 = (x + width * cos + length * sin, y + width * sin - length * cos)
    x3, y3 = (x - width * cos + length * sin, y - width * sin - length * cos)
    x4, y4 = (x - width * cos - length * sin, y - width * sin + length * cos)
    return Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


def compute_collision(pose1,pose2):
    # from l5kit.evaluation import metrics as l5metrics
    # from l5kit.planning import utils

    pred_centroid=pose1[:2]
    pred_yaw=pose1[2]
    pred_extent=pose1[3:]

    p1 = _get_bounding_box(centroid=pred_centroid, yaw=pred_yaw, extent=pred_extent)

    pred_centroid=pose2[:2]
    pred_yaw=pose2[2]
    pred_extent=pose2[3:]

    p2 = _get_bounding_box(centroid=pred_centroid, yaw=pred_yaw, extent=pred_extent)

    collision = p1.intersects(p2)

    return collision

class MyClosedLoopEvaluator:

    def __init__(self, visualizer,verbose=False):
        self.scene_fraction=0.8

        self.max_distance_ref_trajectory=2

        self.reset()

        self.verbose=verbose

        self.max_acc=3

        self.step_time=0.4

        self.visualizer = visualizer

        self.type_length_array=np.array([0,12.5,5,12.5,5.83,2.5,5])
        self.type_width_array=np.array([0,4,2,3.33,2.67,1,2])/2

        self.set_up=False

        # self.type_length_array=np.array([0,10,5,10,5,2,5])#{' Bus':1, ' Car':2, ' Heavy Vehicle':3, ' Medium Vehicle':4, ' Motorcycle':5, ' Taxi':6
        # self.type_width_array=np.array([0,3,2,3,2,1,2])#{' Bus':1, ' Car':2, ' Heavy Vehicle':3, ' Medium Vehicle':4, ' Motorcycle':5, ' Taxi':6


    def setup(self,device,manager):
        self.set_up=True

        self.all_point = torch.FloatTensor(manager.all_point[:,:2]).to(device)

        self.all_point_width = torch.FloatTensor(manager.all_point[:,2]+1.5).to(device)

        segment = torch.FloatTensor(manager.segment).to(device)

        self.edge_lengths =torch.FloatTensor(manager.edge_len).to(device)
        self.segment_width = segment[None, :, 4]
        self.segment_id = segment[:, 5].to(int)

        p2 = segment[:, :2]  # 3,None,2
        p3 = segment[:, 2:4]  # 3,None,2
        self.vector = p3 - p2
        self.distance_sq = torch.linalg.norm(self.vector, axis=-1)[None] ** 2
        self.n = int(torch.max(self.segment_id)) + 1
        self.p2=p2[None]

        self.type_length=torch.FloatTensor(self.type_length_array).to(device)
        self.type_width=torch.FloatTensor(self.type_width_array).to(device)#{' Bus':1, ' Car':2, ' Heavy Vehicle':3, ' Medium Vehicle':4, ' Motorcycle':5, ' Taxi':6

        try:
            self.real_edge_data = np.load("data/meta/macro_results/real_macro_data.npy")  # match point
        except:
            from .real_longterm import get_real_macro

            self.real_edge_data=get_real_macro(self)

        self.real_edge_data=torch.FloatTensor(self.real_edge_data).to(device)

    def reset(self):
        self.off_ref=0

        self.jerk=[]
        self.acc=[]

        self.pos_se_sum=0
        self.pos_num_sum=0

        self.pos_se_sum_min=0
        self.pos_num_sum_min=0

        self.speed_se_sum=0
        self.speed_num_sum=0

        self.dist =[]
        self.off_road=[]

        self.discomfort=[]

        self.min_ade_list=[]

        self.road_density_rmses=[]

        self.road_speed_rmses=[]


    def validate(self):

        res_agg = {}
        # res_agg["distance_ref_trajectory"] = torch.cat(self.dist, dim=0).mean()
        res_agg["pos_rmse"] =torch.sqrt(self.pos_se_sum/self.pos_num_sum).mean()
        res_agg["speed_rmse"] =torch.sqrt(self.speed_se_sum/self.speed_num_sum).mean()

        res_agg["min_ade"] =torch.cat(self.min_ade_list, dim=0).mean()

        res_agg["off_road"] =torch.cat(self.off_road, dim=0).mean()
        # res_agg["discomfort"] =torch.cat(self.discomfort, dim=0).to(float).mean()

        self.reset()

        return res_agg

    def get_edge_id(self,point_array, length_array,speed_array):
        data = torch.zeros([self.n, 3]).to(point_array.device)

        if len(point_array):

            p1 = point_array[:, None]  # None,4,2

            vector_p = p1 - self.p2

            u = torch.einsum("abi,bi->ab", vector_p, self.vector) / self.distance_sq

            closest_points = self.p2 + u[:, :, None] * self.vector[None]

            distances = torch.linalg.norm(p1 - closest_points, dim=-1)

            distances[(u > 1) | (u < 0)] = 1000

            ratio = distances / self.segment_width

            min_id = torch.argmin(ratio, dim=-1)

            min_ratio = torch.amin(ratio, dim=-1) < 1

            nearest_edge_id = self.segment_id[min_id]

            nearest_edge_id = nearest_edge_id[min_ratio]
            length_array = length_array[min_ratio]
            speed_array=speed_array[min_ratio]

            edge_id, count = torch.unique(nearest_edge_id, return_counts=True)

            data[edge_id, 0] = count.to(torch.float)

            data[:, 1]=scatter_add(length_array, nearest_edge_id,out=data[:, 1])
            data[:, 2]=scatter_add(speed_array, nearest_edge_id,out=data[:, 2])

        return data

    def get_edge_res(self,state):
        prev_mask=state[:,5].to(bool)

        state=state[prev_mask]

        point_array = state[:, :2]
        type_array = state[:, 2].to(int)
        prev_point=state[:,3:5]
        velocity=(point_array-prev_point)/0.4

        speed_array=torch.linalg.norm(velocity,dim=-1)

        length_array = self.type_length[type_array]

        edge_parameter = self.get_edge_id(point_array, length_array,speed_array)

        # off_road = self.get_off_road(point_array)

        #pp_point = state[:, 6:8]
        # pp_mask = state[:, 8].to(bool)
        #
        # prev_velocity = (prev_point - pp_point) / 0.4

        # acc = (velocity - prev_velocity) / 0.4
        #
        # discomfort=torch.linalg.norm(acc[pp_mask],dim=-1)>3

        return edge_parameter#,off_road#,discomfort


    def get_macrometric(self,sim_data,i):

        real_data=self.real_edge_data[i][:len(sim_data)]

        pred_density_data = sim_data[..., 0]/ self.edge_lengths[None]

        real_density_data = real_data[..., 0] / self.edge_lengths[None]

        road_density_rmse=torch.sqrt(torch.square(real_density_data-pred_density_data).mean(-1)).mean(0)

        pred_speed_data=sim_data[...,2]/(sim_data[...,0]+0.00001)

        real_speed_data=real_data[...,2]/(real_data[...,0]+0.00001)

        mask=(sim_data[...,0]>0) & (real_data[...,0]>0)

        error=(torch.square(pred_speed_data-real_speed_data)*mask).sum(1)

        edge_num=torch.sum(mask,dim=1)

        road_speed_rmse=torch.sqrt(error/edge_num)[edge_num>0].mean(0)

        return road_density_rmse,road_speed_rmse

    def compute_micrometric(self,simulation_output,device):
        agent_dataset, real_dataset, type_array, engaged_ids, date_index, real_time = simulation_output

        sim_len = len(real_dataset)
        engaged_ids = np.array(list(engaged_ids))

        simulated_centroid = np.zeros([sim_len, len(engaged_ids), 2])
        observed_centroid = np.zeros([sim_len, len(engaged_ids), 2])

        for t in range(sim_len):
            pos, ids = agent_dataset[t]

            index = np.where(engaged_ids == ids[:, None])

            simulated_centroid[t][index[1]] = pos[index[0]]

            off_road = self.get_off_road(torch.FloatTensor(pos).to(device))

            self.off_road.append(off_road)

            pos, ids = real_dataset[t]

            index = np.where(engaged_ids == ids[:, None])

            observed_centroid[t][index[1]] = pos[index[0]]

        # d=np.linalg.norm(simulated_centroid-observed_ego_states,axis=-1)
        if self.visualizer is not None:
            self.visualizer.vis(simulated_centroid, type_array[engaged_ids], date_index, real_time)

        simulated_centroid = torch.FloatTensor(simulated_centroid).to(device)
        observed_centroid = torch.FloatTensor(observed_centroid).to(device)

        sim_mask = simulated_centroid[:, :, 0] != 0
        obs_mask = observed_centroid[:, :, 0] != 0
        mask = sim_mask & obs_mask

        position_se = torch.square(simulated_centroid - observed_centroid).sum(-1)

        self.pos_se_sum += torch.sum(position_se * mask, dim=1)

        self.pos_num_sum += torch.sum(mask, dim=1)

        sim_vel = (simulated_centroid[1:] - simulated_centroid[:-1]) / 0.4

        obs_vel = (observed_centroid[1:] - observed_centroid[:-1]) / 0.4

        speed_mask = mask[1:] & mask[:-1]

        speed_se = torch.square(sim_vel - obs_vel).sum(-1)

        self.speed_se_sum += torch.sum(speed_se * speed_mask, dim=1)

        self.speed_num_sum += torch.sum(speed_mask, dim=1)

        return position_se,mask

    def longterm_eval(self,edge_data,i,device):

        edge_res_list,simulation_output=edge_data

        sim_data=torch.stack(edge_res_list,dim=0)

        self.compute_micrometric( simulation_output, device)

        road_density_rmse,road_speed_rmse=self.get_macrometric(sim_data,i)

        self.road_density_rmses.append(road_density_rmse)

        self.road_speed_rmses.append(road_speed_rmse)


    def longterm_validate(self):

        longterm_results = {
            "longterm_pos_rmse": torch.sqrt(self.pos_se_sum/self.pos_num_sum).mean(),
            "longterm_speed_rmse":torch.sqrt(self.speed_se_sum/self.speed_num_sum).mean(),
            "longterm_offroad":torch.cat(self.off_road, dim=0).mean(),
            "road_density_rmse": torch.stack(self.road_density_rmses).mean()*1000,
            "road_speed_rmse": torch.stack(self.road_speed_rmses).mean(),
        }

        self.reset()

        return longterm_results

    def get_off_road(self,point_array):

        distance=torch.linalg.norm(point_array[:, None]-self.all_point[None],dim=-1)

        distance=distance-self.all_point_width[None]

        min_distance=torch.amin(distance,dim=1)

        off_road=(min_distance > 0).to(float)

        return off_road


    def evaluate(self, sim_outs_list,device):

        scene_num=len(sim_outs_list[0])

        roll_num=len(sim_outs_list)

        for i in range(scene_num):

            ade_list=[]
            mask_list=[]

            for j in range(roll_num):

                simulation_output=sim_outs_list[j][i]

                position_se,mask=self.compute_micrometric(simulation_output,device)

                ade_list.append(torch.sqrt(position_se))

                mask_list.append(mask)

                # sim_acc=(sim_vel[1:]-sim_vel[:-1])/0.4
                #
                # sim_speed_mask=sim_mask[1:]&sim_mask[:-1]
                #
                # acc_mask=sim_speed_mask[1:]&sim_speed_mask[:-1]
                #
                # acc=torch.linalg.norm(sim_acc[acc_mask],dim=-1)
                #
                # discomfort= acc>3
                #
                # self.discomfort.append(discomfort)

                # real_speed_mask=obs_mask[1:]&obs_mask[:-1]
                #
                #
                # if len(engaged_ids):
                #     self.real_collision.append(self.get_collision(observed_centroid,real_speed_mask,obs_vel,type_array,engaged_ids))
                #
                #     self.collision.append(self.get_collision(simulated_centroid,sim_speed_mask,sim_vel,type_array,engaged_ids))

                # simulated_fraction_length = int(len(observed_centroid) * self.scene_fraction)
                #
                # euclidean_distance = torch.linalg.norm(simulated_centroid[:simulated_fraction_length,None] - observed_centroid, ord=2, dim=-1)
                #
                # lat_distance=torch.amin(euclidean_distance, dim=1)
                #
                # lat_distance=lat_distance[sim_mask[:simulated_fraction_length]]
                #
                # self.dist.append(lat_distance)

            whole_mask= mask_list[0]

            for mask in mask_list:
                whole_mask=whole_mask&mask

            all_avail=torch.all(whole_mask,dim=0)

            ade=torch.stack(ade_list,dim=0)[:,:,all_avail]

            ade=torch.mean(ade,dim=1)

            min_ade=torch.amin(ade,dim=0)

            self.min_ade_list.append(min_ade)


    def get_collision(self,pos_all,mask,vel,type_array,engaged_ids):

        pos=pos_all[1:].cpu().numpy()
        heading=torch.atan2(vel[:,:,1],vel[:,:,0])#-pi,pi
        mask=mask&(heading!=0)
        mask=mask.cpu().numpy()
        heading=heading.cpu().numpy()

        length=self.type_length_array[type_array[engaged_ids]]
        width=self.type_width_array[type_array[engaged_ids]]

        radius=np.sqrt(length**2+width**2)/2

        radius_sum=radius[:,None]+radius[None]

        inter_dist=np.linalg.norm(pos[:,:,None]-pos[:,None],axis=-1)

        possible_collsion_mask=inter_dist<radius_sum

        heading_mask=mask[:,:,None]&mask[:,None]

        possible_collsion_mask=possible_collsion_mask&heading_mask

        triangle_mask=np.tril(np.ones([len(engaged_ids),len(engaged_ids)]),k=-1).astype(bool)

        possible_collsion_mask=possible_collsion_mask&triangle_mask[None]

        indice_t,indice_1,indice_2=np.nonzero(possible_collsion_mask)

        pose=np.concatenate([pos,heading[:,:,None],width[None,:,None].repeat(len(pos),axis=0),length[None,:,None].repeat(len(pos),axis=0)],axis=-1)

        pose1=pose[indice_t,indice_1]
        pose2=pose[indice_t,indice_2]

        with Pool(len(os.sched_getaffinity(0))) as pool:
            result = pool.starmap(compute_collision, zip(pose1,pose2))

        result=np.array(result)

        collision_result=np.zeros_like(possible_collsion_mask)

        collision_result[possible_collsion_mask]=result

        agent_collision_num=np.sum(collision_result,axis=1)+np.sum(collision_result,axis=2)

        agent_collsion=agent_collision_num>0

        collision=agent_collsion[mask]

        return torch.FloatTensor(collision).to(vel.device)
