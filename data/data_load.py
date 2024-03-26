import pandas as pd
import numpy as np

def data_process(dates,edge_list):

    data=[]
    index=[]

    for i in dates:
        print(i)

        df= pd.read_hdf('./data/meta/traj_data/data.h5', key='traj' + str(i), mode='r')
        route_df=pd.read_hdf('./data/meta/traj_data/route.h5', key='route' + str(i), mode='r')

        df=pd.merge(df,route_df,on=['time','track_id'])

        ids = pd.read_hdf('./data/meta/traj_data/data.h5', key='all_id' + str(i), mode='r')

        vehicle_types = [' Car', ' Medium Vehicle', ' Heavy Vehicle', ' Taxi', ' Bus', ' Motorcycle']

        vehicle_ids = ids.loc[ids['type'].isin(vehicle_types)].loc[:, 'track_id'].values

        df = df.loc[df['track_id'].isin(vehicle_ids)]

        df = df[df['route_id'] != 0]

        mapped_types=ids.loc[:,'type'].values.map({' Bus':1, ' Car':2, ' Heavy Vehicle':3, ' Medium Vehicle':4, ' Motorcycle':5, ' Taxi':6,' Pedestrian':7,' Bicycle':8})

        types=mapped_types.categories[mapped_types.codes].asi8

        track_ids=ids.loc[:,'track_id'].values

        goal_list =[]

        route_list=[]

        track_list=[]

        type_list=[]

        max_len=0

        for i,(track_id, gr) in enumerate(df.groupby('track_id')):
            track_type=types[track_ids==track_id][0]

            route=gr.loc[:, 'route_id'].values

            pos = gr.loc[:, ['x','y','speed']].values

            goal=pos[-1]

            goal_list.append(goal)

            unique_route=np.unique(route)

            route_edge=[]

            for edge_id in unique_route:

                edge=edge_list[edge_id]

                route_edge.extend(edge)

            max_len = max(len(route_edge), max_len)

            route_list.append(route_edge)

            track_list.append(track_id)

            type_list.append(track_type)

        df = df.loc[df['track_id'].isin(track_list)]

        goal_array=np.array(goal_list)

        track_array=np.array(track_list)

        type_array=np.array(type_list)

        route_array=np.zeros([len(goal_list),max_len,5])#.astype(np.uint16)

        for i in range(len(goal_list)):
            route_i=np.array(route_list[i])
            route_array[i][:len(route_i)]=route_i

        id_map=np.zeros([max(track_array)+1]).astype(np.uint16)

        id_map[track_array]=np.arange(len(track_array))

        agent_index=[]

        agent_array=[]

        agent_id_array=[]

        for time, gr in df.groupby('time'):

            postions = gr.loc[:, ['x','y']].values

            #route_dist=gr.loc[:,'route_dist'].values
            track_id=gr.loc[:, 'track_id'].values

            track_id = id_map[track_id]

            index_array = np.zeros([3]).astype(np.uint32)

            index_array[0] = len(data)
            index_array[1] = len(agent_index)
            index_array[2] = time

            agent_index.append(len(agent_array))

            agent_array.extend(postions.astype(np.float32))

            #agent_id_array.extend(np.stack([track_id,route_dist],axis=-1))
            agent_id_array.extend(track_id)

            index.append(index_array)

        agent_index.append(len(agent_array))

        agent_array=np.stack(agent_array,axis=0)
        agent_id_array=np.stack(agent_id_array,axis=0)
        agent_index=np.stack(agent_index,axis=0)

        data.append([agent_array,agent_id_array,agent_index.astype(np.uint32),type_array.astype(np.uint8),goal_array.astype(np.float32),route_array.astype(np.float32)])

        # break

    index=np.stack(index)

    return data,index
