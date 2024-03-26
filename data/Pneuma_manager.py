import numpy as np



class PneumaManager():
    def __init__(self,cfg):
        try:
            #edge_list1 = list(np.load("./data/meta/changed_networks/network_3m.npy", allow_pickle=True))

            self.edge_list= list(np.load("./data/meta/networks/network_3m.npy", allow_pickle=True))

            # edge_list=[]
            #
            # for edge in self.edge_list:
            #     x=edge[:-1,0]-edge[1:,0]
            #     y=edge[:-1,1]-edge[1:,1]
            #
            #     heading=np.arctan2(y,x)
            #
            #     heading=np.concatenate([heading,np.zeros([1])],axis=0)
            #
            #     edge_list.append(np.concatenate([edge,heading[:,None]],axis=-1))
            #
            # self.edge_list=edge_list

            #self.edge_list[363]=edge_list1[363]

            #self.edge_list[596]=edge_list1[596]

            #self.edge_list[1642]=edge_list1[1642]

            self.segment=np.load("./data/meta/networks/segment.npy")#changed_

            self.edge_len=np.load("./data/meta/networks/edge_len.npy")#changed_

        except:
            from .utils import net, interpolate,node_polygons_list

            edge_mask = np.load("./data/meta/networks/edge_mask.npy")

            self.edge_list = []

            edge_segment_list = []

            segment_width = []

            segment_id = []

            edge_len=[]

            n = 0

            for i,e in enumerate(net._edges):

                point_linestring = np.array(e.getShape(includeJunctions=False))
                lane_num = len(e._lanes)

                edge_width=lane_num*1.6

                interpolated_edge = interpolate(point_linestring, 3,method="meter")

                interpolated_edge=np.concatenate([interpolated_edge,edge_width+np.zeros([len(interpolated_edge),1])],axis=-1)

                to_node=e._to

                center=to_node._coord[:2]

                node_shape = np.array(to_node._shape)

                width=np.mean(np.linalg.norm(node_shape-center,axis=-1))

                node_pos=np.array([[center[0],center[1],width]])

                interpolated_edge=np.concatenate([interpolated_edge,node_pos],axis=0)

                interpolated_edge=np.concatenate([interpolated_edge,np.zeros_like(interpolated_edge[:,:1])+i],axis=1)

                self.edge_list.append(interpolated_edge)

                if edge_mask[i]:

                    length = 0

                    for j in range(len(point_linestring) - 1):
                        edge_segment_list.append(point_linestring[j:j + 2])
                        segment_width.append(edge_width)
                        segment_id.append(n)

                        length += np.linalg.norm(point_linestring[j + 1] - point_linestring[j])

                    edge_len.append(np.array([length * lane_num]))

                    n += 1

            np.save("./data/meta/networks/network_3m", self.edge_list)

            segment_array = np.array(edge_segment_list)
            segment_width = np.array(segment_width)
            segment_id = np.array(segment_id)
            self.edge_len = np.concatenate(edge_len, axis=0)

            p2 = segment_array[:, 0]  # 3,None,2
            p3 = segment_array[:, 1]  # 3,None,2

            self.segment = np.concatenate([p2, p3, segment_width[:, None], segment_id[:, None]], axis=1)

            np.save("./data/meta/networks/segment", self.segment)
            np.save("./data/meta/networks/edge_len", self.edge_len)

        self.edges = []

        for edge in self.edge_list:
            self.edges.append(np.concatenate([edge[:1], edge[-2:]], axis=0))

        self.edges = np.array(self.edges)[...,:2]

        try:
            self.all_point = np.load("./data/meta/networks/point_3m.npy")
        except:

            all_point_list = []

            for i, edge in enumerate(self.edge_list):
                if edge_mask[i]:
                    all_point_list.extend(edge)

            self.all_point = np.array(all_point_list)

            np.save("./data/meta/networks/point_3m.npy", self.all_point)

        try:
            self.traffic_light = np.load('./data/meta/traffic_light/traffic_light.npy')
        except:
            from .utils import get_lights

            self.traffic_light = get_lights()


