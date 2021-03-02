# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks
    
# The Score of 5*E model
class FiveStarE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(FiveStarE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 8 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        re_head, im_head = lhs[:, :self.rank], lhs[:, self.rank:2*self.rank]
        re_relation_a, im_relation_a, re_relation_b, im_relation_b, re_relation_c, im_relation_c, re_relation_d, im_relation_d = rel[:, :self.rank], rel[:, self.rank:2*self.rank], rel[:, 2*self.rank:3*self.rank], rel[:, 3*self.rank:4*self.rank], rel[:, 4*self.rank:5*self.rank], rel[:, 5*self.rank:6*self.rank], rel[:, 6*self.rank:7*self.rank], rel[:, 7*self.rank:]

        re_tail, im_tail = rhs[:, :self.rank], rhs[:, self.rank:2*self.rank]

        re_score_a = re_head * re_relation_a - im_head * im_relation_a
        im_score_a = re_head * im_relation_a + im_head * re_relation_a

        # ah + b
        re_score_top = re_score_a + re_relation_b
        im_score_top = im_score_a + im_relation_b

        # ch
        re_score_c = re_head * re_relation_c - im_head * im_relation_c
        im_score_c = re_head * im_relation_c + im_head * re_relation_c

        # ch + d
        re_score_dn = re_score_c + re_relation_d
        im_score_dn = im_score_c + im_relation_d

        # (ah + b)Conj(ch+d)
        dn_re = torch.sqrt(re_score_dn * re_score_dn + im_score_dn * im_score_dn)

        up_re = torch.div(re_score_top * re_score_dn + im_score_top * im_score_dn, dn_re)
        up_im = torch.div(re_score_top * im_score_dn - im_score_top * re_score_dn, dn_re)

        return torch.sum(
            (up_re) * re_tail +
            (up_im) * im_tail,
            1, keepdim=True
        )
    
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        re_head, im_head = lhs[:, :self.rank], lhs[:, self.rank:2*self.rank]
        re_relation_a, im_relation_a, re_relation_b, im_relation_b, re_relation_c, im_relation_c, re_relation_d, im_relation_d = rel[:, :self.rank], rel[:, self.rank:2*self.rank], rel[:, 2*self.rank:3*self.rank], rel[:, 3*self.rank:4*self.rank], rel[:, 4*self.rank:5*self.rank], rel[:, 5*self.rank:6*self.rank], rel[:, 6*self.rank:7*self.rank], rel[:, 7*self.rank:]


        re_tail, im_tail = rhs[:, :self.rank], rhs[:, self.rank:2*self.rank]

        re_score_a = re_head * re_relation_a - im_head * im_relation_a
        im_score_a = re_head * im_relation_a + im_head * re_relation_a

        # ah + b
        re_score_top = re_score_a + re_relation_b
        im_score_top = im_score_a + im_relation_b

        # ch
        re_score_c = re_head * re_relation_c - im_head * im_relation_c
        im_score_c = re_head * im_relation_c + im_head * re_relation_c

        # ch + d
        re_score_dn = re_score_c + re_relation_d
        im_score_dn = im_score_c + im_relation_d

        # (ah + b)(ch+d)^-1
        dn_re = torch.sqrt(re_score_dn * re_score_dn + im_score_dn * im_score_dn)
        up_re = torch.div(re_score_top * re_score_dn + im_score_top * im_score_dn, dn_re)
        up_im = torch.div(re_score_top * im_score_dn - im_score_top * re_score_dn, dn_re)

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:2*self.rank]

        return (
            (up_re) @ to_score[0].transpose(0, 1) +
            (up_im) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(re_head ** 2 + im_head ** 2),
            torch.sqrt(re_relation_a ** 2 + im_relation_a ** 2 + re_relation_c ** 2 + im_relation_c ** 2 + re_relation_b ** 2 + im_relation_b ** 2 + re_relation_d ** 2 + im_relation_d ** 2),
            torch.sqrt(re_tail ** 2 + im_tail ** 2)
        )
    
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size,:2*self.rank
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        re_head, im_head = lhs[:, :self.rank], lhs[:, self.rank:2*self.rank]
        re_relation_a, im_relation_a, re_relation_b, im_relation_b, re_relation_c, im_relation_c, re_relation_d, im_relation_d = rel[:, :self.rank], rel[:, self.rank:2*self.rank], rel[:, 2*self.rank:3*self.rank], rel[:, 3*self.rank:4*self.rank], rel[:, 4*self.rank:5*self.rank], rel[:, 5*self.rank:6*self.rank], rel[:, 6*self.rank:7*self.rank], rel[:, 7*self.rank:]

        re_score_a = re_head * re_relation_a - im_head * im_relation_a
        im_score_a = re_head * im_relation_a + im_head * re_relation_a

        # ah + b 
        re_score_top = re_score_a + re_relation_b
        im_score_top = im_score_a + im_relation_b

        # ch
        re_score_c = re_head * re_relation_c - im_head * im_relation_c
        im_score_c = re_head * im_relation_c + im_head * re_relation_c

        # ch + d
        re_score_dn = re_score_c + re_relation_d
        im_score_dn = im_score_c + im_relation_d

        # (ah + b)(ch+d)^-1
        dn_re = torch.sqrt(re_score_dn * re_score_dn + im_score_dn * im_score_dn)
        up_re = torch.div(re_score_top * re_score_dn + im_score_top * im_score_dn, dn_re)
        up_im = torch.div(re_score_top * im_score_dn - im_score_top * re_score_dn, dn_re)

        return torch.cat([
            up_re,
            up_im
        ], 1)


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)
