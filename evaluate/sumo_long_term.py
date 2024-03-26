import traci
import pandas as pd
import numpy as np
import torch


def eval_sumo_longterm(my_evaluator,dataset,device):

    leader_data=[]

    sim_len=2000

    hist_len=10

    for i in range(5):
        print(i)
        date=str(i+15)

        scene_dataset = dataset.get_scene_dataset(dataset.start_index[i], sim_len)

        real_dataset,type_array,goal,route,id = scene_dataset  # not include current step

        df = pd.read_hdf('./data/meta/traj_data/data.h5', key='traj' + date, mode='r')

        time=df["time"].values/1000.0

        time=np.sort(np.unique(time))#20551

        min_time=np.min(time)

        max_time=np.max(time)

        sumoCmd = ["sumo",
                   "-S","True",
                   "-r", "./data/meta/sumo/route/routes" + date + ".rou.xml",
                   "-a", './data/meta/sumo/sumo_fine_types.add.xml',
                   "-n", "./data/meta/networks/"+ date +".net.xml",
                   "-b", str(min_time),  # begin time
                    "-e",str(max_time),
                   "--step-length", "0.4",
                    "-W","true"
                   ]

        traci.start(sumoCmd, port=7911)

        edge_res_list=[]

        length_dict={'car': 5, 'taxi': 5, 'MediumVehicle': 5.83, 'HeavyVehicle': 12.5, 'bus': 12.5, 'motorcycle': 2.5}

        agent_datasets = []

        engaged_ids=set()

        for n,t in enumerate(time):
            if n%10==0:

                traci.simulationStep(t)

                point_array=[]
                length_array=[]
                speed_array=[]
                ids=[]


                for id in  traci.vehicle.getIDList():
                    point_array.append(traci.vehicle.getPosition(vehID=str(id)))
                    speed_array.append(traci.vehicle.getSpeed(vehID=str(id)))
                    length_array.append(length_dict[traci.vehicle.getTypeID(vehID=str(id))])
                    # sumo_acc = traci.vehicle.getAcceleration(str(id))
                    #
                    # discomforts.append(abs(sumo_acc) > 3)

                    id=int(id)

                    ids.append(id)

                    engaged_ids.add(id)
                    leader = traci.vehicle.getLeader(str(id))
                    if leader is not None:
                        leader_id, rel_distance = leader
                        leader_data.append(rel_distance)

                point_array = np.array(point_array)
                length_array = np.array(length_array)
                speed_array = np.array(speed_array)

                agent_datasets.append((point_array,np.array(ids)))

                point_array = torch.FloatTensor(point_array).cuda()
                length_array = torch.FloatTensor(length_array).cuda()
                speed_array = torch.FloatTensor(speed_array).cuda()

                edge_parameter = my_evaluator.get_edge_id(point_array, length_array,speed_array)

                edge_res_list.append(edge_parameter)

                if len(edge_res_list)==sim_len+hist_len:
                    break

        traci.close()

        edge_data =(edge_res_list[hist_len:],(agent_datasets[hist_len:],real_dataset[hist_len:], type_array, engaged_ids,0,0))

        my_evaluator.longterm_eval(edge_data,i,device)

    result=my_evaluator.longterm_validate()

    print(result)

        # road_density_rmse,road_speed_rmse=my_evaluator.get_macrometric(sumo_edge_data,i-15)
        #
        # road_density_rmses.append(road_density_rmse)
        # road_speed_rmses.append(road_speed_rmse)
        #
        # edge_data=sumo_edge_data.cpu().numpy()
        #
        # sumo_marco_data.append(edge_data)
        # engaged_ids=np.array(list(engaged_ids))
        #
        # simulated_centroid = np.zeros([len(agent_datasets), len(engaged_ids), 2])
        #
        # for i in range(len(agent_datasets)):
        #     pos, ids= agent_datasets[i]
        #
        #     if len(pos):
        #
        #         index = np.where(engaged_ids == ids[:, None])
        #
        #         simulated_centroid[i][index[1]] = pos[index[0]]
        #
        # all_sumo_pos.append((simulated_centroid,1))


        # break

    # all_sumo_pos=np.concatenate(all_sumo_pos,axis=1)

    # print("road_density_rmse",torch.tensor(road_density_rmses).mean().item(),
    #       "road_speed_rmse",torch.tensor(road_speed_rmses).mean().item(),
    #      )

    # np.save("./data/meta/macro_results/sumo_leader_data",np.array(leader_data))
    #
    # np.save("./data/meta/macro_results/sumo_macro_data",np.array(sumo_marco_data))

#road_density_rmse 0.05267748981714249 road_speed_rmse 5.51995849609375 discomfort_longterm 0.08054067805871945
# road_density_rmse 0.05267748981714249 road_speed_rmse 5.51995849609375 discomfort_longterm 0.08054067805871945
# np.save("./data/meta/macro_results/sumo_pos_data",np.array(all_sumo_pos))
# "discomfort_longterm",np.array(discomforts).mean().item()
#eval_sumo_longterm()

#{'longterm_pos_rmse': tensor(825.8075, device='cuda:0'), 'longterm_speed_rmse': tensor(11.5691, device='cuda:0'), 'longterm_offroad': tensor(0.0012, device='cuda:0', dtype=torch.float64), 'road_density_rmse': tensor(53.6914, device='cuda:0'), 'road_speed_rmse': tensor(5.8613, device='cuda:0')}
