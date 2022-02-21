""" Product manifold for mixed curvature embeddings """
import numpy as np
import torch
import geoopt as geo
from torch import nn

from models.base import KGModel

MIX_MODELS = ["spheric", "euclidean", "hyperbolic", "mixed"]

class BaseM(KGModel):

    def __init__(self, args):
        super(BaseM, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        # Make model accessible to funtions
        self.model = args.model
        # initialize entity and relation weights
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        # initialize signature
        self.sDims = 1
        self.sCurv = 1
        self.eDims = 1
        self.hDims = 1
        self.hCurv = 1
        self.signature()
        # defines the appropriate manifold
        if(args.model == "spheric"):
            self.embed_manifold = geo.SphereProjection(sCurv)
        elif(args.model == "euclidean"):
            self.embed_manifold = geo.Euclidean()
        elif(args.model == "hyperbolic"):
            self.embed_manifold = geo.PoincareBall(hCurv)
        else:
            self.embed_manifold = geo.ProductManifold((geo.SphereProjection(sCurv), sDims), (geo.Euclidean(eDims)), (geo.PoincareBall(hCurv), hDims))
            
    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if(eval_mode):
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])
            
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        #lhs_e, c = lhs_e
        if(eval_mode):
            score = lhs_e @ rhs_e.transpose(0,1)
            score = torch.empty((len([1 for i in lhs_e]), len([ 1 for i in rhs_e])), dtype=self.data_type, device='cuda:0')
            i = 0
            for coordinate in lhs_e:
                scoreRow = - torch.sum((rhs_e - coordinate).pow(2), dim=1)
                score[i] = scoreRow
                i = i + 1
        else:
            if(self.model == "euclidean"):
                # Score is the negative squared euclidean distance 
                # (geoopt.Euclidean.dist(2) doesn't calculate the correct distance, sum over components is missing)
                score = - torch.sum(self.embed_manifold.dist2(lhs_e, rhs_e), dim=1)
                # Reshape to recover the second dimension, which gets dropped by torch.sum (from torch.size([500]) to torch.size([500,1]))
                score = torch.reshape(score, (len([1 for i in lhs_e]), 1))
            elif(self.model == "spheric"):
                pass
            elif(self.model == "hyperbolic"):
                pass
            else:
                pass
        return score

    def signature(self):
        self.sDims = 10
        self.sCurv = 1
        self.eDims = 10
        self.hDims = 10
        self.hCurv = 1
        
class euclidean(BaseM):
    """Euclidean embedding """

    def get_queries(self, queries):
        head = self.entity(queries[:, 0])
        relation = self.rel(queries[:, 1])
        lhs_e = head + relation
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    
class hyperbolic(BaseM):
    """ Hyperbolic embedding """
    
    def get_queries(self, queries):
        head = self.entity(queries[:, 0])
        relation = self.rel(queries[:, 1])