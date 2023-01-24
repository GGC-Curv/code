# CONGREGATE: Contrastive Graph Clustering in Curvature Space.

## Noteworthy Contribution 

In this paper, we are the FIRST to introduce __a novel curvature space__, supporting fine-grained curvature modeling, to graph clustering.

## Datasets

We evaluate our model on 4 benchmark datasets, and all the datasets are publicly available. Please refer to the following papers for further details on the datasets.

+ Cora and Citeseer.	
* Sen, P.; Namata, G.; Bilgic, M.; Getoor, L.; Gallagher, B.; and Eliassi-Rad, T. 2008. Collective Classification in Network Data. AI Mag., 29(3): 93–106.
* Devvrit, F.; Sinha, A.; Dhillon, I,; and Jain. P. S3GC: Scalable self-supervised graph clustering. In Advances in 36th NeurIPS, 2022.

+ MAG-CS
* Park, N.; Rossi, R.; Koh, E.; Burhanuddin, I. A.; Kim, S.; Du, F.; Ahmed, N. K.; and Faloutsos, C. CGC: Contrastive graph clustering for community detection and tracking. In Proceedings of The ACM Web Conference, pages 1115–1126.  ACM, 2022.

+ Amazon-Photo
* Li, B.; Jing, B.; and Tong, H. Graph communal contrastive learning. In Proceedings of The ACM Web Conference, pages 1203–1213. ACM, 2022.
	

## Implementation 

Our model consists of M restricted manifolds and 1 free manifold.
The number of M, and the dimension of manifolds need to be configured for an specific instantiation. (Note that, the restricted manifolds have learnable curvatures, which is another novelty of our model.)
Also, the weighting coefficient alpha's are the hyperparameters of the loss function.

We give a sample implementation of CONGREGATE here.
We will release all the source code of our project HCS_GC, heterogeneous curvature space-graph clustering after publication. 

The requirements is listed below.
+ Python 3.7
+ Pytorch >= 1.1
+ numpy
+ scikit-learn
+ networkx

## Baselines

We compare with 19 state-of-the-art baselines in total. The baselines are introduced in the Technical Appendix, and all the baselines are implemented according to the original papers. 

Please refer to Technical Appendix for further details.



