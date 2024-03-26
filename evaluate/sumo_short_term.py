import pandas as pd
import numpy as np
import traci


def eval_sumo_shortterm():
    distances=[]
    velocities=[]
    discomforts=[]
    ade=[]

    for i in range(50):
        distances.append([])
        velocities.append([])

    for i in range(15,20):


        print(i)
        df = pd.read_hdf('./data/meta/traj_data/data.h5', key='traj' + str(i), mode='r')
        route_df = pd.read_hdf('./data/meta/traj_data/route.h5', key='route' + str(i), mode='r')
        ids = pd.read_hdf('./data/meta/traj_data/data.h5', key='all_id' + str(i), mode='r')

        df = pd.merge(df, route_df, on=['time', 'track_id'])

        vehicle_types = [' Car', ' Medium Vehicle', ' Heavy Vehicle', ' Taxi', ' Bus', ' Motorcycle']

        vehicle_ids = ids.loc[ids['type'].isin(vehicle_types)].loc[:, 'track_id'].values

        df = df.loc[df['track_id'].isin(vehicle_ids)]

        df=df[df['route_id']!=0]

      #  time = df["time"].values

       # unique_time= np.unique(time)
        #print(np.unique(unique_time[1:]-unique_time[:-1]))

        df=df[df["time"]%400<10]

        df['speed']/=3.6

        track_id = df.loc[:, 'track_id'].values

        not_same_id = (track_id[1:] - track_id[:-1]) != 0

        # same_id=np.concatenate([])traci.exceptions.TraCIException: Invalid type 'passenger' for vehicle '8206'.

        x=df["x"].values
        y=df["y"].values

        vx = (x[1:] - x[:-1]) / 0.4 + not_same_id * 10000
        vy = (y[1:] - y[:-1]) / 0.4 + not_same_id * 1000

        df['vx'] = np.concatenate([vx, np.zeros([1]) + 10000], axis=0)
        df['vy'] = np.concatenate([vy, np.zeros([1]) + 10000], axis=0)

        types = ids.loc[:, 'type'].values
        track_ids = ids.loc[:, 'track_id'].values

        types = types.map(
            {' Bus': "bus", ' Car': "car", ' Heavy Vehicle': "HeavyVehicle",
             ' Medium Vehicle': "MediumVehicle", ' Motorcycle': "motorcycle", ' Taxi': "taxi"})


        sim_len=50


        for n, (t, gr) in enumerate(df.groupby('time')):

            step=n % (sim_len+1)

            if step == 0:

                sumoCmd = ["sumo",
                           "-S", "True",
                           "-r", "./data/meta/sumo/route/routes" + str(i) + ".rou.xml",
                           "-a", './data/meta/sumo/sumo_fine_types.add.xml',
                           "-n", "./data/meta/networks/" + str(i) + ".net.xml",
                           "-b", str(t/ 1000.0+0.01),  # begin time
                           "--step-length", "0.4",
                           "-W", "true"
                           ]

                traci.start(sumoCmd)#, port=7911+i)

                xs = gr.loc[:, 'x'].values
                ys = gr.loc[:, 'y'].values
                speeds= gr.loc[:,'speed'].values#Speed is in km/h
                track_id = gr.loc[:, 'track_id'].values

                for x, y,speed, vehID in zip(xs, ys, speeds,track_id):
                    typeID = types[track_ids == vehID][0]

                    traci.vehicle.add(vehID=str(vehID), routeID=str(vehID), typeID=typeID)
                    traci.vehicle.moveToXY(vehID=str(vehID), edgeID="", lane="0", x=str(x),
                                           y=str(y), keepRoute=0)
                    traci.vehicle.setSpeed(vehID=str(vehID), speed=speed)#m/s


            else:
                traci.simulationStep(t/ 1000.0)

                track_id = gr.loc[:, 'track_id'].values

                real_positions = gr.loc[:, ['x','y']].values

                real_velocities = gr.loc[:, ['vx','vy']].values

                # sumo_postions=[]

                sumo_track_id=traci.vehicle.getIDList()

                # print(sumo_track_id)
                # print(track_id)

                #print(traci.simulation.getDepartedIDList())

                for real_pos,real_vel,veh_id in zip(real_positions,real_velocities,track_id):
                    if str(veh_id) in sumo_track_id and real_vel[0]<1000:
                        #print(veh_id,traci.vehicle.getRoute(str(veh_id)))
                        sumo_pos=traci.vehicle.getPosition(str(veh_id))
                        sumo_angle=(90-traci.vehicle.getAngle(str(veh_id)))/180*np.pi#-np.pi#-np.pi#np.pi/2-
                        sumo_speed=traci.vehicle.getSpeed(str(veh_id))

                        sumo_acc=traci.vehicle.getAcceleration(str(veh_id))

                        discomforts.append(abs(sumo_acc)>3)

                        vel_x=np.cos(sumo_angle)*sumo_speed
                        vel_y=np.sin(sumo_angle)*sumo_speed

                        sumo_vel=np.array([vel_x,vel_y])

                        #print(sumo_angle%(np.pi*2),np.arctan2(real_vel[1],real_vel[0])%(np.pi*2))

                        distance_gap=np.square(sumo_pos-real_pos).sum(-1)
                        velocity_gap=np.square(sumo_vel-real_vel).sum(-1)

                        distances[step-1].append(distance_gap)
                        velocities[step-1].append(velocity_gap)

                        ade.append(np.linalg.norm(sumo_pos-real_pos))


                #distances.extend(np.linalg.norm(real_positions-np.array(sumo_postions),axis=-1))

                if step==sim_len:
                    traci.close()
      #              break


        traci.close()
     #   break

    for i in range(50):
        distances[i]=np.sqrt(np.array(distances[i]).mean())
        velocities[i]=np.sqrt(np.array(velocities[i]).mean())

    distances_rmse=np.array(distances).mean()
    velocities_rmse=np.array(velocities).mean()
    ade=np.array(ade).mean()

    print("pos_rmse",distances_rmse,"vel_rmse",velocities_rmse,"ade",ade,"imfeasibility",np.array(discomforts).mean())


eval_sumo_shortterm()
