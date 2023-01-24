import networkx as nx
import torch
from torch_geometric.datasets import Planetoid
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci


class Graph:
    """
    We give a sample for Cora dataset from torch_geometric.datasets
    """

    def __init__(self, dataset='Cora'):
        super(Graph, self).__init__()
        dataset = Planetoid(root='data', name=dataset)
        self.feature_nodes = dataset.data.x
        self.label_nodes = dataset.data.y
        self.edge_index = dataset.data.edge_index
        self.num_node = dataset.data.num_nodes
        self.num_edge = dataset.data.num_edges
        self.num_label = dataset.num_classes
        self.train_mask = dataset.data.train_mask
        self.val_mask = dataset.data.val_mask
        self.test_mask = dataset.data.test_mask
        self.G = nx.Graph()
        self.build_graph()
        self.adj = nx.adjacency_matrix(self.G).todense()

    def build_graph(self):
        edge_index = self.edge_index.T.numpy()
        edges = [tuple(edge) for edge in edge_index]
        self.G.add_edges_from(edges)
        for node in self.G.nodes:
            self.G.nodes[node]['feature'] = self.feature_nodes[node]
            self.G.nodes[node]['label'] = self.label_nodes[node]
        return self.G

    def ricci_curvature(self):
        rc = OllivierRicci(self.G)
        rc.compute_ricci_flow()
        for n1, n2 in self.G.edges():
            self.G.edges[n1, n2]['ricci'] = rc.G.edges[n1, n2]['ricciCurvatur']
        for n in self.G.nodes:
            ri = []
            for e in self.G.edges(n):
                ri.append(self.G.edges[e]['ricci'])
            self.G.nodes[n]['ricci'] = sum(ri) / len(ri)
        return self.G


def get_node_ricci(g):
    ricci_node = []
    for node in g.nodes():
        ricci_node.append(g.nodes[node]['ricci'])
    return torch.tensor(ricci_node)


def get_edge_ricci(g):
    ricci_edge = torch.zeros(len(g.nodes), len(g.nodes))
    for n1, n2 in g.edges():
        ricci_edge[n1, n2] = g.edges[n1, n2]['ricci']
    return ricci_edge


    
