import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperboloid import Hyperboloid


class GraphAttentionLayer(nn.Module):
    """
    This is a sample of tangential method (-fRGCN variant), involving a tangent space.
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.5, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = torch.matmul(a_input, self.a).squeeze(2)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, act=torch.sigmoid):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.act = act

    def forward(self, x, k):
        xt = self.act(self.manifold.logmap0(x, k))
        return self.manifold.expmap0(xt, k)


class GCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.manifold = Hyperboloid()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(self.input_dim - 1, self.output_dim - 1))
        self.att = GraphAttentionLayer(self.output_dim, self.output_dim)
        self.hypact = HypAct(manifold=self.manifold)
        
    def forward(self, x, adj, k):
        logx = self.manifold.logmap0(x, k)
        
        Wx = torch.matmul(logx[:, 1:], self.W)
        zeros = torch.zeros(x.size(0), 1)
        tmp = torch.cat((zeros, Wx), 1)
        
        fun7 = self.manifold.expmap0(tmp, k)
        fun7 = self.manifold.normalize(fun7, k)
        embeddings = self.att(self.manifold.logmap0(fun7, k), adj)

        embeddings = self.manifold.normalize(embeddings, k)
        res = self.hypact(embeddings, k)
        return res

