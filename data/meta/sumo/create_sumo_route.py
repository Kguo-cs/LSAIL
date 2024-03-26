import h5py
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from pyproj import CRS, Transformer
import os
import pickle
import geopandas as gpd
from shapely.geometry import Point
from multiprocessing import Pool
from leuvenmapmatching.map.sqlite import SqliteMap
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
import sumolib
import xml.etree.ElementTree as ET
from datetime import datetime
import traci
import math

sumoCmd = ["sumo",
           "-S","True",
           "-n", "./data/meta/networks/all.net.xml",
           ]

traci.start(sumoCmd, port=7911)


net1 = sumolib.net.readNet('../networks/all.net.xml')

edge_list=[]
edge_dict={}
edge_geometry_list=[]
edge_from_list=[]
edge_length_list=[]

for i,edge in enumerate(net1.getEdges()):

    edge_id=edge._id

    edge_list.append(edge_id)

    edge_from_list.append(edge._from._id)

    edge_geometry_list.append((edge.getShape(includeJunctions=False),edge._length))


    # n1=node_dict[edge._from._id]
    # n2=node_dict[edge._to._id]


    # edge_dict[(edge._from._id, edge._to._id)]=i

def dist_to_segment(p, a, b):
    """Calculate the distance between point p and line segment [a,b]."""
    ap = p - a
    ab = b - a
    t = ap.dot(ab) / ab.dot(ab)
    if t < 0:
        return math.dist(p, a),0
    elif t > 1:
        return math.dist(p, b),np.linalg.norm(ab)
    else:
        proj = a + t * ab
        return math.dist(p, proj),t*np.linalg.norm(ab)

def get_length(edge_geometry,point):
    edge_geometry,max_length=edge_geometry
    dist_to_segs = []

    lengths = []

    proj_lens = []

    for i_ in range(len(edge_geometry) - 1):
        start_point = np.array(edge_geometry[i_])
        end_point = np.array(edge_geometry[i_ + 1])

        dist_to_seg, proj_len = dist_to_segment(point, start_point, end_point)

        length = np.linalg.norm(end_point - start_point, axis=-1)

        dist_to_segs.append(dist_to_seg)

        lengths.append(length)

        proj_lens.append(proj_len)

    min_index = np.argmin(dist_to_segs)

    length_sum = np.sum(lengths[:min_index]) + proj_lens[min_index]

    return min(length_sum,max_length)


edge_list=np.array(edge_list)

for i in range(15,20):
    print(i)

    df = pd.read_hdf('../traj_data/data.h5', key='traj' + str(i), mode='r')
    ids = pd.read_hdf('../traj_data/data.h5', key='all_id' + str(i), mode='r')

    route_df = pd.read_hdf('../traj_data/route.h5', key='route' + str(i), mode='r')

    df = pd.merge(df, route_df, on=['time', 'track_id'])



    tree = ET.parse('route/no.rou.xml')

    root = tree.getroot()

    types = ids.loc[:, 'type'].values
    types = types.map(
        {' Bus': "bus", ' Car': "car", ' Heavy Vehicle': "HeavyVehicle", ' Medium Vehicle': "MediumVehicle",
         ' Motorcycle': "motorcycle", ' Taxi': "taxi",
         ' Pedestrian': "pedestrian", ' Bicycle': "bicycle"})


    departure_times=[]

    routes=[]

    typeIDs=[]

    for id,(track_id, gr) in enumerate(df.groupby('track_id')):
        typeID=types[id]

        if typeID in ['pedestrian','bicycle']:
            continue

        route_=gr.loc[:,'route_id'].values

        x = gr.loc[:, 'x'].values
        y = gr.loc[:, 'y'].values
        times=gr.loc[:, 'time'].values/1000

        dist_mask=route_!=0

        index_true = np.nonzero(dist_mask)[0]

        if len(index_true):
            first_index = index_true[0]

            last_index = index_true[-1] + 1

            edge_array=route_[first_index:last_index].astype(int)

            route_list=[(-1,0)]
            node_list=[]

            for t,edge_id in enumerate(edge_array):
                if edge_id != route_list[-1][0]:

                    node_from=edge_from_list[edge_id]

                    node_list.append(node_from)
                    route_list.append((edge_id,t))


            route_list=np.array(route_list)[1:,0]

            departure_time=times[first_index]

            # pos=route_[-4:]

            start_x=x[first_index]

            start_y=y[first_index]

            end_x = x[last_index-1]

            end_y = y[last_index-1]

            #edge_geometry_list.append()


            edgeId, laneposition, laneIndex =traci.simulation.convertRoad(start_x,start_y)

            end_edgeId, end_laneposition, end_laneIndex =traci.simulation.convertRoad(end_x,end_y)


            # if edgeId!=edge_list[start_edge]:
            # print(track_id,edgeId,edge_list[route_list[0]])



            if edgeId!=edge_list[route_list[0]]:
                #print(laneposition,laneposition0)
                edge_geometry = edge_geometry_list[route_list[0]]
                laneposition=get_length(edge_geometry,np.array([start_x, start_y]))


            if end_edgeId!=edge_list[route_list[-1]]:
                # print(end_laneposition,laneposition1)
                 edge_geometry = edge_geometry_list[route_list[-1]]
                 end_laneposition = get_length(edge_geometry, np.array([end_x, end_y]))

            # print(edgeId,edge_list[route_list[0]],laneposition0,laneposition)
            #
            # print(end_edgeId,edge_list[route_list[-1]],laneposition1,end_laneposition)


            # if end_edgeId!=edge_list[route_list[-1]]:
            #     print(track_id,end_edgeId,edge_list[route_list[-1]])
            #
            route_edges=edge_list[route_list]

            routes.append(route_edges)#(route_edges,laneposition, laneIndex,end_laneposition, end_laneIndex))

            typeIDs.append((typeID,track_id,str(laneposition),str(end_laneposition)))

            departure_times.append(departure_time)


    sort_index=np.argsort(np.array(departure_times))

    for index in sort_index:

        departure_time=departure_times[index]
        route_edges=routes[index]
        typeID,track_id,laneposition,end_laneposition=typeIDs[index]

        vehicle_elem = ET.SubElement(root, 'vehicle', id=str(track_id), type=typeID, depart=str(departure_time),
                                    departPos=laneposition, departLane="free", arrivalPos=end_laneposition)

        route_elem = ET.SubElement(vehicle_elem, 'route', edges=" ".join(route_edges))


        route_id = ET.SubElement(root, 'route', id=str(track_id), edges=" ".join(route_edges))


    tree.write('./data/meta/sumo/route/routes'+str(i)+'.rou.xml')

    # break

    # write the updated XML file
    #tree.write('allpoint100000.add.xml', encoding='UTF-8', xml_declaration=True)
