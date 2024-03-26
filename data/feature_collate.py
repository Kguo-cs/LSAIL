import torch
import numpy as np


class TensorFeatureCollate:
    def __call__(self, batch):

        agent_polylines_list=[]
        target_positions_list=[]

        max_len=0

        for data in batch:
            agent_polylines_list.append(data["agent_polylines"])
            target_positions_list.append(data["positions"])
            if len(data["agent_polylines"])>max_len:
                max_len=len(data["agent_polylines"])

        for i in range(len(batch)):
            agent_polylines=agent_polylines_list[i]
            target_positions=target_positions_list[i]
            device=agent_polylines.device

            agent_polylines_list[i]=torch.cat([agent_polylines,torch.zeros([max_len-len(agent_polylines),agent_polylines.shape[1]],device=device)],dim=0)
            target_positions_list[i]=torch.cat([target_positions,torch.zeros([max_len-len(target_positions),target_positions.shape[1],target_positions.shape[2]],device=device)],dim=0)

        agent_polylines=torch.stack(agent_polylines_list,dim=0)

        target_positions=torch.stack(target_positions_list,dim=0)

        if "value_advs" in batch[0].keys():
            value_advs_list = []
            for data in batch:
                value_advs=data["value_advs"]
                value_advs_list.append( torch.cat([value_advs, torch.zeros(
                    [max_len - len(value_advs), value_advs.shape[1]],device=value_advs.device)], dim=0))

            value_advs=torch.stack(value_advs_list,dim=0)

            return agent_polylines,target_positions,value_advs[...,0],value_advs[...,1]

        else:
            return agent_polylines,target_positions


class FeatureCollate:
    def __call__(self, batch):

        agent_polylines_list=[]

        max_len=0


        for data in batch:
            agent_polylines_list.append(data["agent_polylines"])
            if len(data["agent_polylines"])>max_len:
                max_len=len(data["agent_polylines"])

        for i in range(len(batch)):
            agent_polylines=agent_polylines_list[i]
            agent_polylines_list[i]=np.concatenate([agent_polylines,np.zeros([max_len-len(agent_polylines),agent_polylines.shape[1]])],axis=0)

        agent_polylines=torch.FloatTensor(np.stack(agent_polylines_list,axis=0))

        new_batch = {"agent_polylines": agent_polylines}


        target_positions_list = []

        if "positions" in batch[0].keys():
            for data in batch:
                target_positions=data["positions"]

                pad_positions=np.concatenate([target_positions,np.zeros([max_len-len(target_positions),target_positions.shape[1],target_positions.shape[2]])],axis=0)

                target_positions_list.append(pad_positions)


            target_positions=torch.FloatTensor(np.stack(target_positions_list,axis=0))

            new_batch["positions"]=target_positions

        time_list=[]

        if "next_rel_time" in batch[0].keys():
            for data in batch:
                time_list.append(data["next_rel_time"])

            next_real_time = torch.FloatTensor(np.stack(time_list, axis=0))

            new_batch["next_rel_time"]=next_real_time

        return new_batch
