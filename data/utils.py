import numpy as np
from leuvenmapmatching.map.sqlite import SqliteMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
import pandas as pd
import sumolib
from shapely.geometry import Point,LineString,MultiPolygon,Polygon
import geopandas as gpd
from multiprocessing import Pool
from itertools import repeat
import os
import pyproj
import csv
from tqdm import tqdm

transformer = pyproj.Proj(projparams="+proj=utm +zone=34 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")


net = sumolib.net.readNet('./data/meta/networks/all.net.xml', withLatestPrograms=True)

map_con = SqliteMap("athensmap", use_latlon=False)

edge_polygons_list=[]
edge_directions=[]
node_polygons_list=[]

edge_dict={}
node_dict={}
node_pos_list=[]
edge_id_dict={}
traffic_light_polygons=[]
connections_list=[]
incoming_to_junction_dict={}
tl_list=[]

tl_dict={}

for i,geometry in enumerate(net.getGeometries(useLanes=False,includeJunctions=True)):
    id, point_linestring, width = geometry

    edge = LineString(point_linestring)

    buffer = edge.buffer(width / 2)

    # boundary = buffer.exterior

    edge_polygons_list.append(buffer)

    edge_direction = np.arctan2(point_linestring[-1][1] - point_linestring[0][1],
                                point_linestring[-1][0] - point_linestring[0][0])

    edge_directions.append(edge_direction)

    edge_id_dict[id]=i


for i,node in enumerate(net._nodes):

    node_shape=node._shape
    node_shape.append(node_shape[0])

    poly = Polygon(node_shape)

    node_polygons_list.append(poly)
    x,y,z=node._coord

    map_con.add_node(i,(x,y) )
    node_dict[node._id]=i
    node_pos_list.append(np.array([x,y]))

    if node._type=="traffic_light":
        traffic_light_polygons.append(poly)
        connections=[]

        for incoming in node._incoming:
            connections.append(edge_id_dict[incoming._id])
            incoming_to_junction_dict[edge_id_dict[incoming._id]]=node._id

        connections_list.append(connections)

        tl_dict[node._id]=len(tl_list)

        tl_list.append(node._id)  # 103 light

for i ,edge in enumerate(net.getEdges()):
    n1 = node_dict[edge._from._id]
    n2 = node_dict[edge._to._id]

    map_con.add_edge(n1, n2)
    edge_dict[(n1, n2)] = i

edge_polygons=np.array(edge_polygons_list)
edge_directions=np.array(edge_directions)
traffic_light_polygons = gpd.GeoSeries(traffic_light_polygons)
node_postions=np.array(node_pos_list)
edges_global=MultiPolygon(polygons=edge_polygons_list+node_polygons_list)


def get_minDistance(x, y):
    min_distance = edges_global.distance(Point(x, y))
    return min_distance

def get_route(traj):

    id,x,y,dist=traj
    max_d = 11

    fail = 1
    std_noise = 2

    index_true = np.nonzero(dist==0)[0]#<5

    if len(index_true):

        # Find the index of the last occurrence of True
        first_index = index_true[0]

        last_index = index_true[-1]+1

        x=x[first_index:last_index]
        y=y[first_index:last_index]

        path=list(zip(x, y))

        while True:
            try:
                matcher = DistanceMatcher(map_con, max_dist_init=max_d, max_dist=max_d,
                                      non_emitting_states=False, obs_noise=std_noise)
                states, idx = matcher.match(path, unique=False)
                if  len(states) != len(path):
                    raise Exception("At least one datapoint not mapped or more states than datapoints")
                break
            except:
                fail = fail + 1
                max_d+=fail*fail*fail

        route=np.zeros([len(states)]).astype(int)

        node_list=[(states[0][0],0)]

        node_polygons_route=[node_polygons_list[states[0][0]]]

        for t,state in enumerate(states):
            route[t]=edge_dict[state]
            state_start=state[0]

            if state_start!=node_list[-1][0]:
                node_list.append((state_start,t))
                node_polygons_route.append(node_polygons_list[state_start])

                for n in range(3):
                    prev = n + 3

                    if len(node_list) > prev and node_list[-1][0] == node_list[-prev][0]:

                        inter_node = np.array(node_list[-n - 2:-1])[:, 0]
                        inter_node_pos = node_postions[inter_node]

                        node_dist = np.linalg.norm(inter_node_pos - node_postions[state_start], axis=-1).min()

                        t1 = node_list[-prev][1]
                        t3 = node_list[-1][1]

                        traj = np.stack([x[t1:t3], y[t1:t3]], axis=-1)

                        min_dist = np.linalg.norm(inter_node_pos[:, None] - traj[None], axis=-1).min()
                        #print(min_dist, node_dist, n)

                        if min_dist > node_dist / 2 or node_dist < 15:
                            t2 = (t1 + t3) // 2

                            route[t1:t2] = route[t1 - 1]
                            route[t2:t3] = route[t3]

                            node_list=node_list[:-n-2]


        route_polygons=list(edge_polygons[np.unique(route)])+node_polygons_route
        route_polygons=MultiPolygon(polygons=route_polygons)

        dists_to_edge=[]#np.zeros_like(route)

        for x_pos,y_pos in zip(x, y):

            dist_to_edge=route_polygons.distance(Point(x_pos,y_pos))

            dists_to_edge.append(dist_to_edge)

        route=np.stack([np.array(dists_to_edge),route],axis=-1)

        route=np.concatenate([np.zeros([first_index,2]),route,np.zeros([len(dist)-last_index,2])])

    else:
        route=np.zeros([len(dist),2])

    return route.astype(np.uint16)

def get_light_point(i,df,all_points):

    print(i)

    for id, (track_id, gr) in enumerate(df.groupby('track_id')):

        route_ = gr.loc[:, ['route_dist', 'route_id']].values

        pos_time = gr.loc[:, ['x', 'y', 'time']].values
        speed = gr.loc[:, 'speed'].values
        # dist = route[id][:,0]#distance to route polygon
        edge_id = route_[:, 1]

        # next_edge_id=np.zeros_like(edge_id)
        prev_edge_id = np.zeros_like(edge_id)

        diffs = np.diff(edge_id)

        indices = np.where(diffs != 0)[0] + 1

        for t in range(len(indices)):
            indice = indices[t]

            if t == len(indices) - 1:
                prev_edge_id[indice:] = edge_id[indice - 1]
            else:
                prev_edge_id[indice:indices[t + 1]] = edge_id[indice - 1]
        #
        next_speed = np.concatenate([speed[1:], np.zeros(1)])
        prev_speed = np.concatenate([np.zeros(1), speed[:-1]])

        avail = (speed == 0) & ((next_speed > 0) | (prev_speed > 0)) & (
                    edge_id != 0)  # &(next_edge_id!=0)&(current_dist_stop<10)&(dist<10)

        dates = np.zeros_like(edge_id) + i

        point_x = np.concatenate(
            [pos_time, dates[:, None], prev_speed[:, None], next_speed[:, None], prev_edge_id[:, None],
             edge_id[:, None]], axis=-1)

        avial_point = point_x[avail]

        for point in avial_point:

            point_junction_d = traffic_light_polygons.distance(Point(point[0], point[1]))

            min_d = np.min(point_junction_d)

            if min_d < 15:

                min_id = np.argmin(point_junction_d)

                prev_edge_id = point[-2]
                edge_id = point[-1]

                if edge_id in connections_list[min_id] and edge_polygons_list[int(edge_id)].distance(
                        Point(point[0], point[1])) < 20:
                    point[-2] = edge_id  # +next_edge_id*0.0001                    point[-2] = track_id
                    point[-1] = min_id
                    all_points.append(point)
                elif prev_edge_id in connections_list[min_id] and edge_polygons_list[int(prev_edge_id)].distance(
                        Point(point[0], point[1])) < 20:
                    point[-2] = prev_edge_id  # +edge_id*0.0001                    point[-2]=track_id
                    point[-1] = min_id
                    all_points.append(point)

def get_all_cost(turn_time,cycle_time,time_to_green=None,react_time=2):
    if time_to_green==None:
        time_to_green = np.arange(0, cycle_time, 0.01)  # .repeat(start_num).reshape(-1,start_num)#4500,3

    if len(turn_time) > 1:

        time_diff = turn_time - time_to_green[:, None]

        diff = np.abs(time_diff - np.round(time_diff / cycle_time) * cycle_time)

        mask = diff <= react_time

        arr = np.round(time_diff / cycle_time)

        max_integer = np.amax(np.where(mask, arr, -np.inf), axis=-1)

        min_integer = np.amin(np.where(mask, arr, np.inf), axis=-1)

        integer_range = np.maximum(max_integer - min_integer, 0)

        integer_num = np.sum(mask, axis=-1)

        miss_integer = integer_range - integer_num + 1  # same part miss

        diff_sum = np.sum(diff, axis=-1)

        cost_sum = diff_sum + 1000 * miss_integer - 1000 * integer_num
    else:
        cost_sum=np.ones_like(time_to_green)*10000
    return  cost_sum

def get_cost(possible_time,turn_times,start_point_num):
    cycle_time, start_num, gap, gap1, gap2, group=possible_time
    tl_offset = []
    cost = 0
    for t, (all_move_time, all_stop_time) in enumerate(turn_times):
        if start_num == 1:
            move_time = np.concatenate(np.array(all_move_time, dtype=object)[group]).astype(float)

            all_cost = get_all_cost(move_time, cycle_time)

        elif start_num == 2:
            # turn_time=np.concatenate(connection_turn_time[relation])
            if start_point_num == 2:
                connection_turn_time_1 = all_move_time[0]
                connection_turn_time_2 = all_move_time[1]
            else:
                connection_turn_time_1 = np.concatenate(np.array(all_move_time, dtype=object)[group[0]]).astype(
                    float)
                connection_turn_time_2 = np.concatenate(np.array(all_move_time, dtype=object)[group[1]]).astype(
                    float)

            all_cost0 = get_all_cost(connection_turn_time_1, cycle_time)
            all_cost1 = get_all_cost(connection_turn_time_2 - gap, cycle_time)
            all_cost = all_cost0 + all_cost1
        elif start_num == 3:
            all_cost0 = get_all_cost(all_move_time[0], cycle_time)
            all_cost1 = get_all_cost(all_move_time[group[0]] - gap, cycle_time)
            all_cost2 = get_all_cost(all_move_time[group[1]] - gap1, cycle_time)

            all_cost = all_cost0 + all_cost1 + all_cost2

        else:
            all_cost0 = get_all_cost(all_move_time[0], cycle_time)
            all_cost1 = get_all_cost(all_move_time[group[0]] - gap, cycle_time)
            all_cost2 = get_all_cost(all_move_time[group[1]] - gap1, cycle_time)
            all_cost3 = get_all_cost(all_move_time[group[2]] - gap2, cycle_time)

            all_cost = all_cost0 + all_cost1 + all_cost2 + all_cost3

        min_cost = np.min(all_cost)

        offset = np.arange(0, cycle_time, 0.01)[np.argmin(all_cost)]

        tl_offset.append(offset)

        cost += min_cost

    tl_offset.append(cost)

    return tl_offset

def compute_cost(possible_times,turn_times,start_point_num):


    with Pool(len(os.sched_getaffinity(0))) as pool:

        results= pool.starmap(get_cost,zip(possible_times,repeat(turn_times),repeat(start_point_num)))

    results=np.array(results)
    costs=results[:,-1]
    tl_programs=results[:,:-1]

    min_index = np.argmin(costs)

    cycle_time, start_num, gap,gap1,gap2, group = possible_times[min_index]  # +min_cycle_time

    tl_offset=tl_programs[min_index]

    return cycle_time, start_num, gap,gap1,gap2, group,tl_offset

def get_turn_time(unique_start_point,points):
    turn_times=[]

    for date in np.arange(1,20):
        connection_move_time = []
        connection_stop_time = []

        for start in unique_start_point:

            change_points = points[(points[:, 3] == date)&(points[:, -2] == start)]#.astype(int)

            move_points=change_points[change_points[:,5]>0]

            time = np.sort(move_points[:, 2])

            time = np.concatenate([np.zeros([1]), time])

            time_gap = time[1:] - time[:-1]  # gap_to_prev
            # #
            # # #turn_time = time[1:-1][(time_gap[:-1] > 7) & (time_gap[1:] < 4)]  # later gap
            # #
            time = time[1:][(time_gap > 7) ]#& (time_gap[1:] < 4)  # later gap

            stop_points=change_points[change_points[:,4]>0]

            stop_time= np.sort(stop_points[:, 2])

            stop_time = np.concatenate([np.zeros([1]), stop_time])

            time_gap = stop_time[1:] - stop_time[:-1]  # gap_to_prev

            stop_time = stop_time[1:][(time_gap > 7) ]#& (time_gap[1:] < 4)  # later gap


            connection_move_time.append(time)
            connection_stop_time.append(stop_time)

        turn_times.append((connection_move_time,connection_stop_time))

    return turn_times

def compute_cycle(points):#possible_cycle=

    unique_start_point = np.unique(points[:,-2].astype(int))#,return_counts=True

    turn_times=get_turn_time(unique_start_point,points)

    possible_time=[]

    start_point_num=len(unique_start_point)
    interval = 1

    for cycle_time in [45,90]:#[3]:
        for start_num in range(1,start_point_num+1):
            if start_num==1:
                gap=0
                gap1 = 0
                gap2=0
                if start_point_num>0:
                    possible_time.append((cycle_time, start_num, gap,gap1,gap2,[0]))
                    if start_point_num>1:
                        possible_time.append((cycle_time, start_num, gap,gap1,gap2,[1]))
                    if start_point_num>2:
                        possible_time.append((cycle_time, start_num, gap,gap1,gap2,[2]))
                        # possible_time.append((cycle_time, start_num, gap,gap1,gap2,[0,1]))
                        # possible_time.append((cycle_time, start_num, gap,gap1,gap2,[0,2]))
                        # possible_time.append((cycle_time, start_num, gap,gap1,gap2,[1,2]))
                        # possible_time.append((cycle_time, start_num, gap,gap1,gap2,[0,1,2]))
            elif start_num==2:#and start_point_num==2
                gap1 = 0
                gap2=0

                for gap in np.arange(15,cycle_time-14,interval):
                    if start_point_num==2:
                        possible_time.append((cycle_time,start_num,gap,gap1,gap2,[[0],[1]]))
                    elif start_point_num==3:
                        possible_time.append((cycle_time,start_num,gap,gap1,gap2,[[1,2],[0]]))
                        possible_time.append((cycle_time,start_num,gap,gap1,gap2,[[0,2],[1]]))
                        possible_time.append((cycle_time,start_num,gap,gap1,gap2,[[0,1],[2]]))

                        possible_time.append((cycle_time,start_num,gap,gap1,gap2,[[1],[0]]))#uncontrolled 2
                        possible_time.append((cycle_time,start_num,gap,gap1,gap2,[[1],[2]]))#uncontrolled 0
                        possible_time.append((cycle_time,start_num,gap,gap1,gap2,[[0],[2]]))#uncontrolled 1
            elif start_num==3:
                gap2=0

                for gap in np.arange(15,cycle_time-29,interval):
                    for gap1 in np.arange(gap+15,cycle_time-14,interval):
                        possible_time.append((cycle_time, start_num, gap,gap1,gap2,[1,2]))
                        possible_time.append((cycle_time, start_num, gap,gap1,gap2,[2,1]))
            elif start_num==4:

                for gap in np.arange(15,cycle_time-44,interval):
                    for gap1 in np.arange(gap+15,cycle_time-29,interval):
                        for gap2 in np.arange(gap+30,cycle_time-14,interval):
                            possible_time.append((cycle_time, start_num, gap, gap1,gap2,[1,2,3]))
                            possible_time.append((cycle_time, start_num, gap, gap1,gap2,[2,1,3]))
                            possible_time.append((cycle_time, start_num, gap, gap1,gap2,[1,3,2]))
                            possible_time.append((cycle_time, start_num, gap, gap1,gap2,[2,3,1]))
                            possible_time.append((cycle_time, start_num, gap, gap1,gap2,[3,2,1]))
                            possible_time.append((cycle_time, start_num, gap, gap1,gap2,[3,1,2]))


    cycle_time, start_num, gap,gap1,gap2, group, tl_offset=compute_cost(possible_time,turn_times,start_point_num)

    if start_num==1:
        costs=[]
        for gap in np.arange(cycle_time//2,cycle_time-9,interval):
            cost=0
            for t,(all_move_time,all_stop_time) in enumerate(turn_times):
                offset=np.array([tl_offset[t]])
                stop_time=np.concatenate(np.array(all_stop_time,dtype=object)[group]).astype(float)

                cost+=get_all_cost(stop_time-gap,cycle_time,offset)
            costs.append(cost)

        min_index=np.argmin(costs)

        gap=np.arange(cycle_time//2,cycle_time-9,interval)[min_index]

    return cycle_time, start_num, gap, gap1,gap2,group, tl_offset,unique_start_point

def get_cycle(key,all_points):
    junction_id=tl_list[int(key)]

    points = all_points[all_points[:, -1] == key]

    cycle_time, start_num, gap, gap1, gap2,group, tl_offset,unique_start_point=compute_cycle(points)

    start_point_num=len(unique_start_point)

    print(junction_id, start_point_num,  cycle_time, start_num, gap, gap1,gap2, group, len(points))  # ,start_num, time_gaps,phase_num[junction_id],result#

    result=(junction_id, unique_start_point,  cycle_time, start_num, gap, gap1, gap2,group,tl_offset)

    return result

def get_light(item,date,traffic_light_array,yellow_time = 5):
    junction_id, start_points, cycle_time, start_num, gap, gap1, gap2, group, tl_offset = item

    tls = net.getTLS(junction_id)

    programs = tls._programs['0']

    offset = tl_offset[date - 1]

    programs._offset = round(offset)

    link_list = []

    for start_point in start_points:
        links = []
        for connection in tls._connections:
            inLane, outLane, linkNo = connection

            inLane_edge = edge_id_dict[inLane._edge._id]

            if inLane_edge == start_point:
                links.append(linkNo)

        link_list.append(links)

    if start_num == 1:

        for edge_id in start_points[group]:
            traffic_light_array[date][edge_id][0] = offset
            traffic_light_array[date][edge_id][1] = cycle_time
            traffic_light_array[date][edge_id][2] = gap

        state0 = ['O'] * len(tls._connections)

        for start in group:
            for link_no in link_list[start]:
                state0[link_no] = 'g'

        state0 = ''.join(state0)

        new_phase1 = sumolib.net.Phase(duration=gap - yellow_time, state=state0)

        new_phase2 = sumolib.net.Phase(duration=yellow_time, state=state0.replace('g', 'y'))

        new_phase3 = sumolib.net.Phase(duration=cycle_time - gap, state=state0.replace('g', "r"))

        programs._phases = [new_phase1, new_phase2, new_phase3]

    elif start_num == 2:

        for edge_id in start_points[group[0]]:
            traffic_light_array[date][edge_id][0] = offset
            traffic_light_array[date][edge_id][1] = cycle_time
            traffic_light_array[date][edge_id][2] = gap

        for edge_id in start_points[group[1]]:
            traffic_light_array[date][edge_id][0] = offset + gap
            traffic_light_array[date][edge_id][1] = cycle_time
            traffic_light_array[date][edge_id][2] = cycle_time - gap

        state0 = ['O'] * len(tls._connections)

        for start in group[1]:
            for link_no in link_list[start]:
                state0[link_no] = 'r'

        for start in group[0]:
            for link_no in link_list[start]:
                state0[link_no] = 'g'

        state0 = ''.join(state0)

        new_phase1 = sumolib.net.Phase(duration=gap - yellow_time, state=state0)

        new_phase2 = sumolib.net.Phase(duration=yellow_time, state=state0.replace('g', 'y'))

        state0 = ['O'] * len(tls._connections)

        for start in group[0]:
            for link_no in link_list[start]:
                state0[link_no] = 'r'

        for start in group[1]:
            for link_no in link_list[start]:
                state0[link_no] = 'g'

        state0 = ''.join(state0)

        new_phase3 = sumolib.net.Phase(duration=cycle_time - gap - yellow_time, state=state0)

        new_phase4 = sumolib.net.Phase(duration=yellow_time, state=state0.replace('g', 'y'))

        programs._phases = [new_phase1, new_phase2, new_phase3, new_phase4]

    else:
        edge_id = start_points[0]

        traffic_light_array[date][edge_id][0] = offset
        traffic_light_array[date][edge_id][1] = cycle_time
        traffic_light_array[date][edge_id][2] = gap

        edge_id = start_points[group[0]]

        traffic_light_array[date][edge_id][0] = offset + gap
        traffic_light_array[date][edge_id][1] = cycle_time
        traffic_light_array[date][edge_id][2] = gap1 - gap

        edge_id = start_points[group[1]]

        traffic_light_array[date][edge_id][0] = offset + gap1
        traffic_light_array[date][edge_id][1] = cycle_time
        traffic_light_array[date][edge_id][2] = cycle_time - gap1

        if start_num == 4:
            traffic_light_array[date][edge_id][2] = gap2 - gap1

            edge_id = start_points[group[2]]

            traffic_light_array[date][edge_id][0] = offset + gap2
            traffic_light_array[date][edge_id][1] = cycle_time
            traffic_light_array[date][edge_id][2] = cycle_time - gap2

        state0 = ['r'] * len(tls._connections)

        for link_no in link_list[0]:
            state0[link_no] = 'g'

        state0 = ''.join(state0)

        new_phase1 = sumolib.net.Phase(duration=gap - yellow_time, state=state0)

        new_phase2 = sumolib.net.Phase(duration=yellow_time, state=state0.replace('g', 'y'))

        state0 = ['r'] * len(tls._connections)

        for link_no in link_list[group[0]]:
            state0[link_no] = 'g'

        state0 = ''.join(state0)

        new_phase3 = sumolib.net.Phase(duration=gap1 - gap - yellow_time, state=state0)

        new_phase4 = sumolib.net.Phase(duration=yellow_time, state=state0.replace('g', 'y'))

        state0 = ['r'] * len(tls._connections)

        for link_no in link_list[group[1]]:
            state0[link_no] = 'g'

        state0 = ''.join(state0)

        new_phase5 = sumolib.net.Phase(duration=cycle_time - gap1 - yellow_time, state=state0)

        new_phase6 = sumolib.net.Phase(duration=yellow_time, state=state0.replace('g', 'y'))

        programs._phases = [new_phase1, new_phase2, new_phase3, new_phase4, new_phase5, new_phase6]

        if start_num == 4:
            new_phase5 = sumolib.net.Phase(duration=gap2 - gap1 - yellow_time, state=state0)

            state0 = ['r'] * len(tls._connections)

            for link_no in link_list[group[2]]:
                state0[link_no] = 'g'

            state0 = ''.join(state0)

            new_phase7 = sumolib.net.Phase(duration=cycle_time - gap2 - yellow_time, state=state0)

            new_phase8 = sumolib.net.Phase(duration=yellow_time, state=state0.replace('g', 'y'))

            programs._phases = [new_phase1, new_phase2, new_phase3, new_phase4, new_phase5, new_phase6, new_phase7,
                                new_phase8]

    return tls.toXML()


def get_routes( i, df):
    xs = df['x'].values
    ys = df['y'].values

    with Pool(len(os.sched_getaffinity(0))) as pool:
        result = pool.starmap(get_minDistance, zip(xs, ys))

    df2 = pd.DataFrame(columns=['distances'])

    df2['distances'] = np.array(result).astype(np.float16)

    print("Finish computing distance",i)

    df = pd.concat([df, df2], axis=1)

    traj_list = []

    for id,(track_id, gr) in enumerate(df.groupby('track_id')):

        lat = gr.loc[:, 'x'].values
        lon = gr.loc[:, 'y'].values
        dists = gr.loc[:, 'distances'].values

        traj_list.append((id, lat, lon, dists))

    with Pool(len(os.sched_getaffinity(0))) as pool:
        result = pool.starmap(get_route, zip(traj_list))

    data = []

    for k, (track_id, gr) in enumerate(df.groupby('track_id')):
        times = gr.loc[:, 'time'].values[:, None]

        data.append(np.concatenate([times, track_id + np.zeros_like(times), result[k]], axis=-1))

    route_df = pd.DataFrame(data=np.concatenate(data), columns=['time', 'track_id', 'route_dist', 'route_id'])

    route_df["track_id"] = route_df["track_id"].astype(np.uint16)
    route_df["route_dist"] = route_df["route_dist"].astype(np.uint8)
    route_df["route_id"] = route_df["route_id"].astype(np.uint16)

    route_df.to_hdf('./data/meta/traj_data/route.h5', key='route' + str(i))

    print("Finish computing route",i)

    return route_df


def get_data_from_csv():
    transformer = pyproj.Proj(projparams="+proj=utm +zone=34 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    time_id=0

    try:
        for key in ['1024', '1029', '1030', '1101']:

            dir = "./data/" + key

            for file in os.listdir(dir):
                if file[-3:] == "csv":
                    path_trajectory_data = dir + '/' + file
                    print(path_trajectory_data)

                    val = csv.reader(open(path_trajectory_data, 'r'), delimiter=';')
                    header = next(val)
                    header = [i.strip() for i in header]
                    data = pd.DataFrame(columns=header[:4])
                    df_all = []
                    for line in tqdm(val):
                        data.loc[len(data)] = line[:4]
                        _ = line.pop()
                        tmp_df = pd.DataFrame(
                            {j: pd.Series(line[(i + 4)::6], dtype='float64') for i, j in enumerate(header[4:])})
                        tmp_df.insert(0, 'track_id', [int(line[0])] * len(tmp_df))
                        df_all.append(tmp_df)
                    ids = data.astype(
                        {'track_id': 'int64', 'type': 'category', 'traveled_d': 'float64', 'avg_speed': 'float64'})

                    val = pd.concat(df_all, axis=0)
                    val.loc[:, 'time'] = val.loc[:, 'time'].values * 1000
                    val = val.astype({'time': 'int64'})

                    df_id = pd.DataFrame(columns=['track_id', 'type'])
                    df_id['track_id'] = ids.loc[:, 'track_id'].values.astype(np.uint16)
                    df_id['type'] = ids.loc[:, 'type'].values

                    df_id.to_hdf('./data/meta/traj_data/data.h5', key='all_id' + str(time_id), format="table")

                    lons = val.loc[:, 'lon'].values
                    lats = val.loc[:, 'lat'].values

                    xs, ys = transformer.transform(lons, lats)

                    df = pd.DataFrame(columns=['track_id', 'x', 'y', 'speed', 'time'])

                    df['track_id'] = val.loc[:, 'track_id'].values.astype(np.uint16)
                    df['x'] = xs - 738450.81
                    df['y'] = ys - 4206052.77
                    df['speed'] = val.loc[:, 'speed'].values
                    df['time'] = val.loc[:, 'time'].values.astype(np.uint32)

                    df.to_hdf('./data/meta/traj_data/data.h5', key='traj' + str(time_id))

                    time_id+=1


    except:
        print("download the csv data from https://open-traffic.epfl.ch/")



def get_lights():
    all_points = []

    for i in range(20):

        try:
            df = pd.read_hdf('./data/meta/traj_data/data.h5', key='traj' + str(i), mode='r')
        except:
            get_data_from_csv()
            df = pd.read_hdf('./data/meta/traj_data/data.h5', key='traj' + str(i), mode='r')

        route_df = get_routes(i, df)

        merge_df = pd.merge(df, route_df, on=['time', 'track_id'])

        get_light_point(i, merge_df, all_points)

    all_points = np.stack(all_points)

    all_points[:, 2] /= 1000  # all_points[all_points[:,5]>0]

    junction_key = np.unique(all_points[:, -1])  # 96 light  junction point

    junction_ids = []
    light = []

    for key in junction_key:
        if key not in ["6910572379", "5581434217", "5921003579", "J8", "J16",
                       "J11"]:
            junction_ids.append(key)
            light.append(get_cycle(key,all_points))

    for joint in [["163711425#0", "390115233#0", "163711428#0"], ["189099792#0", "222941078#9", "887747035#0"]]:
        points = []
        for edge_id in joint:
            points.append(all_points[all_points[:, -2] == edge_id_dict[edge_id]])

        points = np.concatenate(points)

        cycle_time, start_num, gap, gap1, gap2, group, tl_offset, unique_start_point = compute_cycle(
            points)

        light.append((
                     incoming_to_junction_dict[unique_start_point[0]], np.array([unique_start_point[0]]), cycle_time, 1,
                     gap, 0, 0, [0], tl_offset))
        light.append((incoming_to_junction_dict[unique_start_point[group[0]]], np.array([unique_start_point[group[0]]]),
                      cycle_time, 1, gap1 - gap, 0, 0, [0], np.array(tl_offset) + gap))
        light.append((incoming_to_junction_dict[unique_start_point[group[1]]], np.array([unique_start_point[group[1]]]),
                      cycle_time, 1, cycle_time - gap1, 0, 0, [0], np.array(tl_offset) + gap1))

        print(cycle_time, start_num, gap, gap1, group)


    traffic_light_array = np.zeros([20, len(net._edges), 3])  # offset, cycle time, green time

    for date in range(1, 20):
        light_xml = '<additionals>\n'

        for item in light:
            light_xml += get_light(item, date, traffic_light_array)

        with open("./data/meta/traffic_light/" + str(date) + ".tll.xml", "w") as f:
            f.write(light_xml)
            f.write("\n")
            f.write("</additionals>")

    traffic_light_array = (traffic_light_array * 1000).astype(np.uint32)

    np.save('./data/meta/traffic_light', traffic_light_array)

    return traffic_light_array


def interpolate(xy: np.ndarray, step,method="step") -> np.ndarray:
    """
    Interpolate points based on cumulative distances from the first one. Two modes are available:
    INTER_METER: interpolate using step as a meter value over cumulative distances (variable len result)
    INTER_ENSURE_LEN: interpolate using a variable step such that we always get step values
    Args:
        xyz (np.ndarray): XYZ coords
        step (float): param for the interpolation
        method (InterpolationMethod): method to use to interpolate

    Returns:
        np.ndarray: the new interpolated coordinates
    """
    cum_dist = np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))
    cum_dist = np.insert(cum_dist, 0, 0)

    if method == "step":
        step = int(step)
        assert step > 1, "step must be at least 2 with INTER_ENSURE_LEN"
        steps = np.linspace(cum_dist[0], cum_dist[-1], step)
    else:
        assert step > 0, "step must be greater than 0 with INTER_FIXED"
        steps = np.arange(cum_dist[0], cum_dist[-1], step)

    steps[-1]=cum_dist[-1]


    xy_inter = np.empty((len(steps), 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=cum_dist, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=cum_dist, fp=xy[:, 1])
    return xy_inter

