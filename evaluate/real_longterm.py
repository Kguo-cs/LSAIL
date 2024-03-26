import numpy as np
import pandas as pd
import torch


def get_real_macro(my_evaluator):

    real_marco_data=[]


    for i in range(15,20):
        print(i)
        df = pd.read_hdf('./data/meta/traj_data/data.h5', key='traj'+str(i), mode='r')
        ids = pd.read_hdf('./data/meta/traj_data/data.h5', key='all_id'+str(i), mode='r')
        route_df = pd.read_hdf('./data/meta/traj_data/route.h5', key='route' + str(i), mode='r')

        df = pd.merge(df, route_df, on=['time', 'track_id'])

        vehicle_types = [' Car', ' Medium Vehicle', ' Heavy Vehicle', ' Taxi', ' Bus',' Motorcycle']

        vehicle_ids = ids.loc[ids['type'].isin(vehicle_types)].loc[:, 'track_id'].values

        df = df.loc[df['track_id'].isin(vehicle_ids)]

        df = df.loc[df['route_id']!=0]

        track_ids = ids.loc[:, 'track_id'].values
        mapped_length = ids.loc[:, 'type'].values.map(
            {' Bus': 12.5, ' Car': 5, ' Heavy Vehicle': 12.5, ' Medium Vehicle': 5.83, ' Motorcycle': 2.5, ' Taxi': 5,
             ' Pedestrian': 7, ' Bicycle': 8})

        #time=df["time"].values


        df['speed'] =df.loc[:, 'speed'].values/3.6

        lengths_data=np.zeros([np.max(track_ids)+1])

        lengths_data[track_ids]=mapped_length.values

        edge_data=[]

        for i,(t, gr) in enumerate(df.groupby('time')):
            if i%10==0:
                # print(t/1000.0)
                xs = gr.loc[:, 'x'].values
                ys = gr.loc[:, 'y'].values
                track_id = gr.loc[:, 'track_id'].values

                length_array=lengths_data[track_id]

                speed_array=gr.loc[:,'speed'].values


                # for x, y, vehID in zip(xs, ys, track_id):
                #     traci.vehicle.moveToXY(vehID=str(vehID), edgeID="", lane="0", x=str(x),  y=str(y), keepRoute=2)
                #
                # traci.simulationStep()

                # edge_result = traci.edge.getAllSubscriptionResults()
                #
                # edge_parameter_list=[]
                #
                # for edge in edge_result.keys():
                #
                #     edge_parameter_list.append([edge_result[edge][traci.constants.LAST_STEP_VEHICLE_NUMBER],edge_result[edge][traci.constants.LAST_STEP_MEAN_SPEED],edge_result[edge][traci.constants.LAST_STEP_LENGTH]])

                # edge_parameter = np.array(edge_parameter_list)

                point_array=np.stack([xs,ys],axis=-1)
                #edge_parameter1=get_edge_id(point_array,length_array)

                point_array=torch.FloatTensor(point_array).cuda()
                length_array=torch.FloatTensor(length_array).cuda()
                speed_array=torch.FloatTensor(speed_array).cuda()

                edge_parameter=my_evaluator.get_edge_id(point_array,length_array,speed_array)
                # edge_parameter=edge_parameter


                # number=edge_parameter[:,0]
                #
                # print(number[number!=data]-data[number!=data])
                # if not np.all(edge_parameter==edge_parameter):
                #     print(np.where(edge_parameter!=edge_parameter))

                edge_data.append(edge_parameter)

                if len(edge_data)==2000:
                    break


        edge_data=torch.stack(edge_data).cpu().numpy()

        real_marco_data.append(edge_data)

    real_marco_data=np.array(real_marco_data)

    np.save("./data/meta/macro_results/real_marco_data", real_marco_data)

    return real_marco_data


# print(edge,edge_result[edge][traci.constants.LAST_STEP_VEHICLE_NUMBER])
