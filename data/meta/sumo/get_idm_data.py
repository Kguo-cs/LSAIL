import traci
import pandas as pd
import numpy as np



def get_data(date_list):
    leader_data=[]
    for i in date_list:
        print(i)
        df = pd.read_hdf('../traj_data/data.h5', key='traj' + str(i), mode='r')
        ids = pd.read_hdf('../traj_data/data.h5', key='all_id' + str(i), mode='r')
        route_df = pd.read_hdf('../traj_data/route.h5', key='route' + str(i), mode='r')

        df = pd.merge(df, route_df, on=['time', 'track_id'])

        vehicle_types = [' Car', ' Medium Vehicle', ' Heavy Vehicle', ' Taxi', ' Bus',' Motorcycle']

        vehicle_ids = ids.loc[ids['type'].isin(vehicle_types)].loc[:, 'track_id'].values

        df = df.loc[df['track_id'].isin(vehicle_ids)]

        speed= df.loc[:, 'speed'].values/3.6
        df['speed'] =speed
        track_id=df.loc[:, 'track_id'].values

        not_same_id=(track_id[10:]-track_id[:-10])!=0

        #same_id=np.concatenate([])

        acc=(speed[10:]-speed[:-10])/0.4+not_same_id*1000

        df['acc']=np.concatenate([acc,np.zeros([10])+1000],axis=0)
       # df=df[same_id==0]


        df = df.loc[df['route_id']!=0]

        types = ids.loc[:, 'type'].values
        track_ids = ids.loc[:, 'track_id'].values
        types = types.map(
            {' Bus': "bus", ' Car': "car", ' Heavy Vehicle': "HeavyVehicle", ' Medium Vehicle': "MediumVehicle",
             ' Motorcycle': "motorcycle", ' Taxi': "taxi",
             ' Pedestrian': "pedestrian", ' Bicycle': "bicycle"})


        time=df["time"].values

        min_time=np.min(time)/1000
        max_time=np.max(time)/1000


        sumoCmd = ["sumo",
                   "-S","True",
                   "-a", './type.add.xml',
                   "-n", "../networks/all.net.xml",
                   "-b", str(min_time),  # begin time
                    "-e",str(max_time),
                   "--step-length", "0.4",
                   "--time-to-teleport", "-1",
                   "--collision.action", "none",
                   "--default.emergencydecel", "-1",
                   "-W", "true"
                   ]

        traci.start(sumoCmd)

        for typeID, vehID in zip(types, track_ids):
            traci.vehicle.add(vehID=str(vehID), routeID='', typeID=typeID)
            traci.vehicle.setSpeedMode(str(vehID), 0)  # strictly follow speed control commands
            traci.vehicle.setLaneChangeMode(str(vehID), 0)  # disable auto lane change
            traci.vehicle.moveToXY(vehID=str(vehID), edgeID="", lane="0", x=str(0),
                                   y=str(0), keepRoute=2)  # , angle=str(phi_sumo),

        # sumo_data=[]
        #
        # edge_data=[]
        # length_dict={'passenger': 5, 'taxi': 5, 'MediumVehicle': 5.83, 'HeavyVehicle': 12.5, 'bus': 12.5, 'motorcycle': 2.5}
        type_dict={'passenger': 0, 'taxi': 1, 'MediumVehicle': 2, 'HeavyVehicle': 3, 'bus': 4, 'motorcycle': 5}

        # for i,(t, gr) in enumerate(df.groupby('track_id')):
        #
        #     acc=

        for i,(t, gr) in enumerate(df.groupby('time')):
            if i%10==0:
                #print(t/1000.0)
                postions = gr.loc[:, ['x','y']].values
                #ys = gr.loc[:, 'y'].values
                speeds=gr.loc[:,'speed'].values
                accels=gr.loc[:,'acc'].values

                track_id = gr.loc[:, 'track_id'].values

                # for x, y, vehID in zip(xs, ys, track_id):
                #     traci.vehicle.moveToXY(vehID=str(vehID), edgeID="", lane="0", x=str(x),  y=str(y), keepRoute=2)

                for vehID in traci.vehicle.getIDList():
                    if int(vehID) not in track_id:
                        traci.vehicle.moveToXY(vehID=vehID, edgeID="", lane="0", x=str(0),
                                               y=str(0), keepRoute=2)  # , angle=str(phi_sumo),
                    else:
                        pos=postions[track_id==int(vehID)][0]
                        traci.vehicle.moveToXY(vehID=vehID, edgeID="", lane="0", x=str(pos[0]),
                                               y=str(pos[1]), keepRoute=2)  # , angle=str(phi_sumo),

                # print(len(traci.vehicle.getIDList()),len(track_id))

                traci.simulationStep()

                # edge_result = traci.edge.getAllSubscriptionResults()
                #
                edge_parameter_list=[]

                # for edge in edge_result.keys():
                #
                #     edge_parameter_list.append([edge_result[edge][traci.constants.LAST_STEP_VEHICLE_NUMBER],edge_result[edge][traci.constants.LAST_STEP_MEAN_SPEED],edge_result[edge][traci.constants.LAST_STEP_LENGTH]])
                #
                # edge_parameter = np.array(edge_parameter_list)

                # no_lead_data=[]



                for id,pos,speed,accel in zip(track_id,postions,speeds,accels):
                    # pos=traci.vehicle.getPosition(vehID=id)
                    #
                    # speed=traci.vehicle.getSpeed(vehID=id)
                    #
                    # acc=traci.vehicle.getAcceleration(vehID=id)
                    #speed=traci.vehicle.getSpeed(str(id))


                    #if speed!=0 and accel>-10 and accel<10:

                    leader=traci.vehicle.getLeader(str(id))
                    #accel=traci.vehicle.getAcceleration(str(id))
                    cur_type = traci.vehicle.getTypeID(str(id))
                    #cur_len = length_dict[cur_type]
                    cur_type_id=type_dict[cur_type]

                    # print(speed, speed_1,acc1,accel)


                    if leader is None:
                        rel_distance=10000
                        rel_speed=0
                        #leader_type_id=0
                    else:
                        leader_id, rel_distance=leader
                        #leader_type=traci.vehicle.getTypeID(leader_id)
                        # leader_speed = traci.vehicle.getSpeed(str(leader_id))
                        leader_speed=speeds[track_id==int(leader_id)][0]

                        # if len(leader_speed)==0:
                        #     leader_speed = traci.vehicle.getSpeed(str(leader_id))

                        #leader_len=length_dict[leader_type]
                       # leader_type_id=type_dict[leader_type]

                        #leader_pos=postions[track_id==int(leader_id)][0]

                        # dist1=dist+(cur_len+leader_len)/2
                        #rel_distance=np.linalg.norm(leader_pos-pos,ord=2)#
                        rel_speed=speed-leader_speed

                        # if rel_distance>0.5:
                    leader_data.append((cur_type_id,speed,rel_speed,accel,rel_distance))
                        #print(cur_type_id,leader_type_id,speed,rel_speed,accel,rel_distance)#front + minGap to the back of the leader

                    #point_array.append(pos)

                    #length_array.append(length_dict[traci.vehicle.getTypeID(vehID=str(id))])

                    #traci.vehicle.ge

                # point_array=np.stack([xs,ys],axis=-1)
                #
                # data=get_edge_id(point_array)
                #
                # number=edge_parameter[:,0]

                #print(number[number!=data]-data[number!=data])

                # edge_data.append(edge_parameter_list)

        #edge_data=np.array(edge_data)

        #sumo_marco_data.append(edge_data)

        traci.close()

    return np.array(leader_data)


# leader_data_val=get_data(range(15,20))
# np.save("idm_data_val",np.array(leader_data_val))
#
# leader_data_train=get_data(range(1,15))
# np.save("idm_train_data",np.array(leader_data_train))

# print(edge,edge_result[edge][traci.constants.LAST_STEP_VEHICLE_NUMBER])
