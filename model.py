import torch
from torch import nn
from torch.nn import functional as F
from layers import GCNLayer
from geoopt import Euclidean, Lorentz, Sphere


class GCNList(nn.Module):

    def __init__(self, feature_dim, embedding_dim, num_manifold):
        super(GCNList, self).__init__()
        self.num_manifold = num_manifold
        self.gcn_list = nn.ModuleList([GCNNet(feature_dim, embedding_dim) for i in range(num_manifold)])

    def forward(self, node_feature, adj, curvatures):
        nodes_embedding = []
        for i in range(self.num_manifold):
            gcn = self.gcn_list[i]
            node_embedding = gcn(node_feature, adj, curvatures[i])
            nodes_embedding.append(node_embedding)
        return torch.stack(nodes_embedding, dim=0)


class GCNNet(nn.Module):
    """
    Note, the parameters living in the manifold are optimized via Riemannian Adam:
                Max Kochurov, Rasul Karimov, and Serge Kozlukov. Geoopt: Riemannian optimization in pytorch. ICML 2020
    """

    def __init__(self, feature_dim, embedding_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNLayer(feature_dim, embedding_dim)
        self.conv2 = GCNLayer(embedding_dim, embedding_dim)

    def forward(self, node_feature, adj, curvature):
        x = self.conv1(node_feature, adj, curvature)
        x = self.conv2(x, adj, curvature)
        return x


class Curvature(nn.Module):
    """
    This class enables the fine-grained curvature modeling of the proposed heterogeneous curvature space,
                    i.e., induce node-level curvature via a mlp
    """

    def __init__(self, ricci_node, num_manifold):
        super(Curvature, self).__init__()
        self.ricci_node = ricci_node
        self.linear1 = nn.Linear(num_manifold, num_manifold)
        self.linear2 = nn.Linear(num_manifold, 1)

    def compute_loss(self, z0, curvatures):
        ricci = self.ricci_mlp(z0, curvatures)
        offset = (self.ricci_node - ricci) ** 2
        return offset

    def ricci_mlp(self, z0, curvatures):
        c0 = z0.split(1, dim=-1)[0]
        cm = curvatures[1:]
        cm = cm.expand(len(c0), len(cm))
        c = torch.cat([c0, cm], dim=-1)
        c = self.linear1(c)
        c = F.leaky_relu_(c)
        c = self.linear2(c)
        return c.squeeze(dim=-1)


class Cluster(nn.Module):

    def __init__(self, ricci):
        super(Cluster, self).__init__()
        self.ricci = ricci

    def compute_loss(self, node_embeddings, community_embedding, alpha, curvatures):
        pi = self.compute_pi(node_embeddings, community_embedding, curvatures)
        loss = alpha * self.intra_loss(pi).sum(-1).mean() - self.inter_loss(pi).sum(-1).mean()
        return loss

    def intra_loss(self, pi: torch.Tensor):
        ijk = []
        for k in range(pi.size(1)):
            row = pi[:, k]
            ij = torch.mm(row.unsqueeze(-1), row.unsqueeze(0))
            ijk.append(ij)
        ijk = torch.stack(ijk, dim=0)
        intra = ijk * self.ricci
        return intra

    def inter_loss(self, pi: torch.Tensor):
        ijk1k2 = []
        for k1 in range(pi.size(1)):
            row1 = pi[:, k1]
            ijk2 = []
            for k2 in range(pi.size(1)):
                row2 = pi[:, k2]
                ij = torch.mm(row1.unsqueeze(-1), row2.unsqueeze(0))
                ijk2.append(ij)
            ijk1k2.append(torch.stack(ijk2, dim=0))
        ijk1k2 = torch.stack(ijk1k2, dim=0)
        inter = ijk1k2 * self.ricci
        return inter

    def compute_pi(self, node_embeddings, community_embedding, curvatures):
        dist = self.dist(node_embeddings, community_embedding, curvatures)
        pi = dist.softmax(dim=-1)
        return pi

    def dist(self, node_embeddings, community_embedding, curvatures):
        dists = []
        for i in range(len(curvatures)):
            node_embedding = node_embeddings[i]
            manifold = get_manifold(curvatures[i])
            dist = manifold.dist(node_embedding.unsqueeze(dim=1), community_embedding.unsqueeze(dim=0))
            dists.append(dist)
        dists = torch.stack(dists, dim=0)
        dist_norm = dists.norm(p=2, dim=0)
        return dist_norm


class Contrastive(nn.Module):

    def __init__(self, embedding_dim, num_manifold):
        super(Contrastive, self).__init__()
        self.linear_log = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(num_manifold - 1)])
        self.linear_sim = nn.Linear(embedding_dim, embedding_dim)
        self.num_manifold = num_manifold

    def forward(self, embeddings, community, curvatures, pi):
        z0 = embeddings[0]
        zm = self.log_map(embeddings[1:], curvatures[1:])
        sim1 = self.similarity(z0, zm)
        coe1 = self.coefficient(z0, zm, pi)
        MI = self.mutual(sim1, coe1)
        pi_c = torch.diag_embed(torch.ones(self.num_manifold))
        sim2 = self.similarity(community, zm)
        coe2 = self.coefficient(community, zm, pi, pi_c)
        MI_c = self.mutual_c(sim2, coe2, pi)
        return MI.mean() + MI_c.mean()

    def log_map(self, embeddings, curvatures):
        embeddings_ = []
        x = torch.zeros(embeddings.size(-1))
        y = torch.zeros_like(x)
        y[0] = 1
        for i in range(self.num_manifold - 1):
            z = embeddings[i]
            manifold = get_manifold(curvatures[i])
            if curvatures[i] > 0:
                z_log = manifold.logmap(y, z)
            else:
                z_log = manifold.logmap(x, z)
            z_ = z * self.linear_log[i](z_log)
            embeddings_.append(z_)
        return torch.stack(embeddings_, dim=0)

    def similarity(self, z0, zm):
        """
        This is a sample of -gLT variant, involving a tangent space.
        """
        sims = []
        for i in range(self.num_manifold - 1):
            z_ = zm[i]
            sim = self.linear_sim(z_).matmul(z0.T)
            sims.append(sim)
        return torch.stack(sims, dim=0)

    def coefficient(self, z0, zm, pi, pi_c=None, beta=2):
        if pi_c is None:
            ijpi = pi.mm(pi.T)
        else:
            ijpi = pi.mm(pi_c.T)
        cosm = []
        for i in range(self.num_manifold - 1):
            z_ = zm[i]
            cos = torch.cosine_similarity(z_.unsqueeze(dim=1), z0.unsqueeze(dim=0), dim=-1)
            cosm.append(cos)
        cosm = torch.stack(cosm, dim=0)
        return torch.abs(ijpi - cosm) ** beta

    def mutual(self, sim, coe):
        MI = []
        prod = sim * coe
        prod_exp = torch.exp(prod)
        for i in range(self.num_manifold - 1):
            p_ = prod_exp[i]
            rate = p_.diag() / p_.sum(dim=-1)
            MI.append(-rate.log())
        return torch.stack(MI, dim=0)

    def mutual_c(self, sim, coe, pi):
        MI = []
        prod = sim * coe
        prod_exp = torch.exp(prod)
        arg = torch.argmax(pi, dim=-1)
        for i in range(self.num_manifold - 1):
            p_ = prod_exp[i]
            p_c = torch.gather(p_, -1, arg.unsqueeze(-1)).squeeze(-1)
            rate = p_c / p_.sum(dim=-1)
            MI.append(-rate.log())
        return torch.stack(MI, dim=0)

def get_manifold(k):
    if k == 0:
        manifold = Euclidean(ndim=1)
    elif k < 0:
        manifold = Lorentz(-k)
    else:
        manifold = Sphere()
    return manifold

  
