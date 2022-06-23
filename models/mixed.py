""" Product manifold for mixed curvature embeddings """
import numpy as np
import torch
import geoopt as geo
from torch import nn

import time

import math

from models.base import KGModel

MIX_MODELS = ["spheric", "euclidean", "hyperbolic", "mixed"]

class BaseM(KGModel):

    def __init__(self, args):
        super(BaseM, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.args = args
        # Make model accessible to funtions
        self.model = args.model
        self.curv = args.curv
        self.curv = 0

        # initialize signature
        if args.model == "mixed":
            #TODO remove manual ratio
            args.non_euclidean_ratio = 0.5
            nonEuclideanDims = math.floor(args.rank * args.non_euclidean_ratio)
            self.sDims = math.floor(nonEuclideanDims * (1 - args.hyperbolic_ratio))
            self.sCurv = args.sphericalCurv
            self.eDims = args.rank - nonEuclideanDims
            self.hDims = nonEuclideanDims - self.sDims
            self.hCurv = args.hyperbolicCurv

        # defines the appropriate manifold
        if self.model == "spheric":
            self.embed_manifold = geo.SphereProjection(learnable=True)
        elif self.model == "euclidean":
            self.embed_manifold = geo.Euclidean(ndim=1)
        elif self.model == "hyperbolic":
            self.embed_manifold = geo.PoincareBall(learnable=True)
        elif self.model == "mixed":
            # For working with the Product manifold
            # All components present
            if self.sDims > 0 and self.eDims > 0 and self.hDims > 0:
                #self.embed_manifold = geo.StereographicProductManifold(
                #    (geo.SphereProjection(learnable=True), self.sDims), (geo.Euclidean(ndim=1), self.eDims), (geo.PoincareBall(learnable=True), self.hDims))
                self.embed_manifold = geo.StereographicProductManifold(
                    (geo.SphereProjection(learnable=True), self.sDims), (geo.PoincareBall(0), self.eDims), (geo.PoincareBall(learnable=True), self.hDims))
            # One component missing
            elif self.sDims == 0 and self.eDims > 0 and self.hDims > 0:
                #self.embed_manifold = geo.StereographicProductManifold(
                #    (geo.Euclidean(ndim=1), self.eDims), (geo.PoincareBall(learnable=True), self.hDims))
                self.embed_manifold = geo.StereographicProductManifold(
                    (geo.PoincareBall(0), self.eDims), (geo.PoincareBall(learnable=True), self.hDims))
            elif self.sDims > 0 and self.eDims == 0 and self.hDims > 0:
                self.embed_manifold = geo.StereographicProductManifold(
                    (geo.SphereProjection(learnable=True), self.sDims), (geo.PoincareBall(learnable=True), self.hDims))
            elif self.sDims > 0 and self.eDims > 0 and self.hDims == 0:
                #self.embed_manifold = geo.StereographicProductManifold(
                #    (geo.SphereProjection(learnable=True), self.sDims), (geo.Euclidean(ndim=1), self.eDims))
                self.embed_manifold = geo.StereographicProductManifold(
                    (geo.SphereProjection(learnable=True), self.sDims), (geo.PoincareBall(0), self.eDims))
            # Two components missing    
            elif self.sDims == 0 and self.eDims == 0 and self.hDims > 0:
                #self.embed_manifold = geo.ProductManifold(
                #    (geo.PoincareBall(self.hCurv), self.hDims))
                self.embed_manifold = geo.StereographicProductManifold(
                    (geo.PoincareBall(learnable=True), self.hDims))
            elif self.sDims == 0 and self.eDims > 0 and self.hDims == 0:
                #self.embed_manifold = geo.StereographicProductManifold(
                #    (geo.Euclidean(ndim=1), self.eDims))
                self.embed_manifold = geo.StereographicProductManifold(
                    (geo.PoincareBall(0), self.eDims))
            elif self.sDims > 0 and self.eDims == 0 and self.hDims == 0:
                self.embed_manifold = geo.StereographicProductManifold(
                    (geo.SphereProjection(learnable=True), self.sDims))

        # Project points onto the manifold
        #self.entity.weight.data = self.embed_manifold.projx(initEntityData)
        #self.rel.weight.data = self.embed_manifold.projx(initRelData)
        
        # initialize diagonal relation matrix (see MuRP)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        #self.rel_diag = nn.Embedding(self.sizes[1], 1)
        #self.rel_diag = nn.Embedding(self.sizes[1], 1)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        #self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 1), dtype=self.data_type) - 1.0
        #self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 1), dtype=self.data_type) - 1.0

        # initialize entity and relation weights
        initEntityData = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        initRelData = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

        self.entity.weight.data = initEntityData
        self.rel.weight.data = initRelData
        if self.model == "TransHyp":
            self.entity.weight.data = geo.ManifoldParameter(initEntityData, self.embed_manifold)
            self.rel.weight.data = geo.ManifoldParameter(initRelData, self.embed_manifold)
            
    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if(eval_mode):
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])
            
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        device = lhs_e.device
        if(eval_mode):
            # create empty tensor, which will later be iteratively filled
            #startTime = time.time()
            score = torch.empty((len([1 for i in lhs_e]), len([ 1 for i in rhs_e])), dtype=self.data_type, device=device)
            i = 0
            for coordinate in lhs_e:
                # score = - distance ** 2
                scoreRow = - self.embed_manifold.dist2(rhs_e, coordinate)
                score[i] = scoreRow
                i = i + 1
        else:
            #startTime = time.time()
            # Score is the negative squared euclidean distance 
            score = - self.embed_manifold.dist2(lhs_e, rhs_e)
            # Reshape to recover the second dimension, which gets dropped by torch.sum (from torch.size([500]) to torch.size([500,1]))
            score = torch.reshape(score, (len([1 for i in lhs_e]), 1))
        return score
        
class euclidean(BaseM):
    """Euclidean embedding """

    def get_queries(self, queries):
        head = self.entity(queries[:, 0])
        rel_diag_matrix = self.rel_diag(queries[:, 1])
        head = head * rel_diag_matrix
        relation = self.rel(queries[:, 1])
        lhs_e = head + relation
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    
class spheric(BaseM):
    """ Spheric embedding """
    
    def get_queries(self, queries):
        head = self.entity(queries[:, 0])
        rel_diag_matrix = self.rel_diag(queries[:, 1])
        head = self.embed_manifold.expmap0(head * rel_diag_matrix)
        relation = self.embed_manifold.expmap0(self.rel(queries[:, 1]))
        lhs_e = self.embed_manifold.mobius_add(head, relation)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    
class hyperbolic(BaseM):
    """ Hyperbolic embedding """
    
    def get_queries(self, queries):
        head = self.entity(queries[:, 0])
        #rel_diag_matrix = torch.block_diag(self.rel_diag(queries[:, 1]))
        rel_diag_matrix = self.rel_diag(queries[:, 1])
        head = self.embed_manifold.expmap0(head * rel_diag_matrix)
        relation = self.embed_manifold.expmap0(self.rel(queries[:, 1]))
        lhs_e = self.embed_manifold.mobius_add(head, relation)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class TransHyp(BaseM):
    """ Hyperbolic embedding """
    
    def get_queries(self, queries):
        head = self.entity(queries[:, 0])
        relation = self.rel(queries[:, 1])
        lhs_e = self.embed_manifold.mobius_add(head, relation)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class mixed(BaseM):
    """ Mixed embedding """
    
    def get_queries(self, queries):
        head = self.entity(queries[:, 0])
        #rel_diag_matrix = self.rel_diag(queries[:, 1])
        #relation = self.embed_manifold.expmap0(self.rel(queries[:, 1]))
        rel_diag_matrix = self.rel_diag(queries[:, 1])
        relation = self.rel(queries[:, 1])

        head = self.embed_manifold.expmap(torch.zeros(head.size(), device=head.device), head * rel_diag_matrix)
        relation = self.embed_manifold.expmap(torch.zeros(relation.size(), device=head.device), relation)

        lhs_e = self.embed_manifold.mobius_add(head, relation)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class mixed_old(BaseM):
    """ Mixed embedding """
    
    def get_queries(self, queries):
        head = self.entity(queries[:, 0])
        #rel_diag_matrix = self.rel_diag(queries[:, 1])
        #head = self.embed_manifold.expmap0(head * rel_diag_matrix)
        #relation = self.embed_manifold.expmap0(self.rel(queries[:, 1]))
        rel_diag_matrix = self.rel_diag(queries[:, 1])
        relation = self.rel(queries[:, 1])
        # Translates the space components separately
        head_comps = []
        rel_comps = []
        for i in range(len(self.components)):
            head_comp = self.embed_manifold.take_submanifold_value(head, i)
            rel_comp = self.embed_manifold.take_submanifold_value(relation, i)
            #print(self.components[i][1])
            #print(self.components)
            #print(self.embed_manifold.manifolds)
            
            if self.components[i][0] == "spheric":
                head_comps.append(geo.SphereProjection(self.components[i][1]).expmap0(head_comp))
                rel_comps.append(geo.SphereProjection(self.components[i][1]).expmap0(rel_comp))
            elif self.components[i][0] == "euclidean":
                head_comps.append(head_comp)
                rel_comps.append(rel_comp)
            elif self.components[i][0] == "hyperbolic":
                head_comps.append(geo.PoincareBall(self.components[i][1]).expmap0(head_comp))
                rel_comps.append(geo.PoincareBall(self.components[i][1]).expmap0(rel_comp))

        head = torch.cat([head_comps[i] for i in range(len(head_comps))], 1)
        relation = torch.cat([rel_comps[i] for i in range(len(rel_comps))], 1)

        lhs_e_comps = []
        for i in range(len(self.components)):
            headComp = self.embed_manifold.take_submanifold_value(head, i)
            relComp = self.embed_manifold.take_submanifold_value(relation, i)
            if self.components[i][0] == "spheric":
                lhs_e_comps.append(geo.SphereProjection(self.components[i][1]).mobius_add(headComp, relComp))
            elif self.components[i][0] == "euclidean":
                lhs_e_comps.append(headComp + relComp)
            elif self.components[i][0] == "hyperbolic":
                lhs_e_comps.append(geo.PoincareBall(self.components[i][1]).mobius_add(headComp, relComp))
        lhs_e = torch.cat([lhs_e_comps[i] for i in range(len(lhs_e_comps))], 1)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases