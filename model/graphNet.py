from torch import nn
import torch
import  numpy as np
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add

def rep2batch(x, batch_shape):
    return torch.repeat_interleave(x[None], int(np.prod(batch_shape)), dim=0).reshape(*batch_shape, *x.shape)

class GraphAttention(nn.Module):
    """
    Graph attention module to pool features from source nodes to destination nodes.

    Given a destination node i, we aggregate the features from nearby source nodes j whose L2
    distance from the destination node i is smaller than a threshold.

    This graph attention module follows the implementation in LaneGCN and is slightly different
    from the one in Gragh Attention Networks.

    Compared to the open-sourced LaneGCN, this implementation omitted a few LayerNorm operations
    after some layers. Not sure if they are needed or not.
    """
    def __init__(self, src_feature_len: int, dst_feature_len: int,dropout:float=0.1):
        super().__init__()
        """
        :param src_feature_len: source node feature length.
        :param dst_feature_len: destination node feature length.
        :param dist_threshold:
            Distance threshold in meters.
            We only aggregate node information if the destination nodes are within this distance
            threshold from the source nodes.
        """
        edge_input_feature_len = src_feature_len + 2+dst_feature_len

        self.edge_attention = nn.Sequential(
            nn.Linear(edge_input_feature_len, 1, bias=False),
            nn.LeakyReLU(0.2),
            # nn.Linear(edge_output_feature_len, 1),
        )

        self.edge_embed = nn.Linear(edge_input_feature_len, dst_feature_len, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        # self.dst_feature_norm = nn.LayerNorm(dst_feature_len)
        #
        #self.output_linear = nn.Linear(dst_feature_len, dst_feature_len)
        self.relu=nn.ReLU()

    def forward(
        self,
        src_node_features: torch.Tensor,
        edge_src_idx: torch.Tensor,
        dst_node_features: torch.Tensor,
        edge_dst_idx: torch.Tensor,
        edge_dist_rel:torch.Tensor
    ) -> torch.Tensor:
        """
        Graph attention module to pool features from source nodes to destination nodes.

        :param src_node_features: <torch.FloatTensor: num_src_nodes, src_node_feature_len>.
            Source node features.
        :param src_node_pos: <torch.FloatTensor: num_src_nodes, 2>. Source node (x, y) positions.
        :param dst_node_features: <torch.FloatTensor: num_dst_nodes, dst_node_feature_len>.
            Destination node features.
        :param dst_node_pos: <torch.FloatTensor: num_dst_nodes, 2>. Destination node (x, y)
            positions.
        :return: <torch.FloatTensor: num_dst_nodes, dst_node_feature_len>. Output destination
            node features.
        """
        # Find (src, dst) node pairs that are within the distance threshold,
        # and they form the edges of the graph.
        # src_dst_dist.shape is (num_src_nodes, num_dst_nodes).

        # src_node_encoded_features = self.src_encoder(src_node_features)
        # dst_node_encoded_features = self.dst_encoder(dst_node_features)
        #
        # # edge_src_features.shape is (num_edges, edge_src_feature_len).
        src_node_expand_features = src_node_features[edge_src_idx]
        dst_node_expand_features = dst_node_features[edge_dst_idx]
        # edge_src_pos.shape is (num_edges, 2).
        # edge_feature = self.edge_dist_encoder(edge_dist_rel)

        edge_input_features = torch.cat([src_node_expand_features, edge_dist_rel, dst_node_expand_features], dim=-1)

        src = self.edge_attention(edge_input_features)[:,0]

        # src_max = torch.scatter_max(src.detach(), edge_dst_idx)
        # out = src - src_max.index_select(0, edge_dst_idx)

        attention=scatter_softmax(src,edge_dst_idx)

        attention=self.dropout(attention)

        edge_input_features=self.edge_embed(edge_input_features)

        weighted_feature=attention[:,None]*edge_input_features

        dst_node_output_features=scatter_add(weighted_feature,edge_dst_idx,dim=0)

        # dst_node_output_features = torch.zeros_like(dst_node_features)#.clone()
        #
        # dst_node_output_features.index_add_(0, edge_dst_idx, weighted_feature)

        dst_node_output_features = self.relu(dst_node_output_features)#+self.output_linear(dst_node_features)

        return dst_node_output_features

class Actor2ActorAttention(nn.Module):
    """
    Actor-to-Actor attention module.
    """

    def __init__(self, actor_feature_len: int, num_attention_layers: int,cfg) -> None:
        """
        :param actor_feature_len: Actor feature length.
        :param num_attention_layers: Number of times to repeatedly apply the attention layer.
        :param dist_threshold_m:
            Distance threshold in meters.
            We only aggregate actor-to-actor node
            information if the actor nodes are within this distance threshold from the other actor nodes.
            The value used in the LaneGCN paper is 30 meters.
        """
        super().__init__()

        self.agent_dist=cfg["agent_dist"]

        self.nearest_number=cfg["nearest_number"]

        dropout=cfg["dropout"]

        attention_layers = [
            GraphAttention(actor_feature_len, actor_feature_len,dropout)
            for _ in range(num_attention_layers)
        ]
        self.attention_layers = nn.ModuleList(attention_layers)

    def edge(self,src_node_pos, dst_node_pos, rotate):

        rel_pos = src_node_pos[:, None] - dst_node_pos[None]

        src_dst_dist = rel_pos.norm(dim=-1)

        nearest_value, indices = torch.topk(src_dst_dist, k=min(len(src_dst_dist), self.nearest_number), dim=0,
                                            largest=False)

        threshold = torch.clamp_max(nearest_value[-1], self.agent_dist)[None]


        # src_dst_dist_mask=src_dst_dist<dist_threshold
        src_dst_dist_mask = src_dst_dist <= threshold  # (src_dst_dist <= threshold[None])&(src_dst_dist>0)#src_dst_dist <= threshold[None]#(src_dst_dist <= threshold[None])&(src_dst_dist>0)##

        rel_pos = torch.einsum("ija,jab->ijb", rel_pos, rotate)

        # edge_src_dist_pairs.shape is (num_edges, 2).
        edge_src_dist_pairs = src_dst_dist_mask.nonzero(as_tuple=False)
        edge_src_idx = edge_src_dist_pairs[:, 0]
        edge_dst_idx = edge_src_dist_pairs[:, 1]
        # edge_src_pos = src_node_pos[edge_src_idx]
        # edge_dst_pos = dst_node_pos[edge_dst_idx]
        # rel_pos1 = edge_src_pos - edge_dst_pos

        edge_feature = rel_pos[src_dst_dist_mask]

        return edge_src_idx, edge_dst_idx, edge_feature

    def forward(
        self,
        actor_features: torch.Tensor,
        src_actor_centers: torch.Tensor,
        dst_actor_centers: torch.Tensor,
        rotate:torch.Tensor,
        rel_relation:torch.Tensor
    ) -> torch.Tensor:
        """
        Perform Actor-to-Actor attention.

        :param actor_features: <torch.FloatTensor: num_actors, actor_feature_len>. Actor features.
        :param actor_centers: <torch.FloatTensor: num_actors, 2>. (x, y) positions of the actors.
        :return: <torch.FloatTensor: num_actors, actor_feature_len>. Actor features after aggregating the lane features.
        """
        if rel_relation is  None:
            actor_src_idx, actor_dis_idx, edge_dist_rel = self.edge(src_actor_centers, dst_actor_centers,rotate)
        else:
            actor_src_idx, actor_dis_idx, edge_dist_rel=rel_relation

        for attention_layer in self.attention_layers:
            actor_features = attention_layer(
                actor_features,
                actor_src_idx,
                actor_features,
                actor_dis_idx,
                edge_dist_rel
            )
        return actor_features,(actor_src_idx, actor_dis_idx, edge_dist_rel)


class GraphNet(nn.Module):

    def __init__(self,cfg,input_dim,output_dim,layer_num=1):
        super(GraphNet, self).__init__()

        d_model = cfg["d_model"]

        self.encoder=nn.Sequential(nn.Linear(input_dim,d_model),nn.LayerNorm(d_model),nn.LeakyReLU(0.2))

        self.graph_layer=Actor2ActorAttention(d_model, layer_num,cfg)

        self.decoder=nn.Linear(d_model,output_dim)

    def forward(
        self,
        state,
        center=None,
        rotate=None,
        rel_relation=None
    ) -> torch.Tensor:

        feature=self.encoder(state)

        feature,rel_relation=self.graph_layer(feature,center,center,rotate,rel_relation)

        feature=self.decoder(feature)

        return feature,rel_relation