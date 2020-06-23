# https://github.com/ValentinFigue/Sklearn_PyTorch/blob/master/Sklearn_PyTorch/binary_tree.py
# 위 코드에 상당한 의존성을 가지고 있음.
# 구체적으로는 안되는 부분을 수정하고 list 자료형을 torch.Tensor로 대체,
# 향후 병렬 처리 및 성능 개선을 실시하여 직접 사용할 예정.

import torch
import torch.nn as nn
import math
import random
from typing import Optional, Tuple, Callable, Union


# utils
def sample_dimensions(
    vectors: torch.Tensor
) -> torch.Tensor:
    length = vectors.size(-1)
    return torch.randint(0, length, size=(int(math.sqrt(length)),))


def sample_vectors(
    vectors: torch.Tensor,
    labels: torch.Tensor, 
    nb_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    length = vectors.size(-1)
    sampled_indices = torch.randint(0, length, size=(nb_samples,))
    sampled_vectors = vectors.index_select(0, sampled_indices)
    sampled_labels = labels.index_select(0, sampled_indices)
    return sampled_vectors, sampled_labels


def divide_set(
    vectors: torch.Tensor, 
    labels: torch.Tensor, 
    column: int, 
    value: Union[int, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vectors_set_1 = []
    vectors_set_2 = []
    values_set_1  = []
    values_set_2  = []
    for vector, label in zip(vectors, labels):
        if vector[column] >= value:
            vectors_set_1.append(vector.unsqueeze(dim=0))
            values_set_1.append(label.unsqueeze(dim=0))
        else:
            vectors_set_2.append(vector.unsqueeze(dim=0))
            values_set_2.append(label.unsqueeze(dim=0))
    vectors_set_1 = torch.cat(vectors_set_1, dim=0) \
                    if not vectors_set_1 == [] else torch.Tensor([])
    vectors_set_2 = torch.cat(vectors_set_2, dim=0) \
                    if not vectors_set_2 == [] else torch.Tensor([])
    values_set_1  = torch.cat(values_set_1, dim=0) \
                    if not values_set_1 == [] else torch.Tensor([])
    values_set_2  = torch.cat(values_set_2, dim=0) \
                    if not values_set_2 == [] else torch.Tensor([])
    return vectors_set_1, values_set_1, vectors_set_2, values_set_2
    
    
# Node Class
class DecisionNode:

    def __init__(
        self, 
        col: int=-1, 
        value: Optional[Union[int, float]]=None, 
        results: Union[int, float]=None, 
        tb: Optional['DecisionNode']=None, 
        fb: Optional['DecisionNode']=None
    ):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        
        
# Decision Tree
class TorchDecisionTreeRegressor(nn.Module):
    
    def __init__(
        self, 
        max_depth: int=-1
    ):
        super().__init__()
        self._root_node = None
        self.max_depth = max_depth

    def fit(
        self, 
        vectors: torch.Tensor, 
        values: torch.Tensor, 
        criterion: Optional[Callable]=None
    ):
        if len(vectors) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if len(vectors) != len(values):
            raise ValueError("Labels and data vectors must have the same number of elements")
        if not criterion:
            criterion = variance
        self._root_node = self._build_tree(
            vectors, values, criterion, self.max_depth)
        

    def _build_tree(
        self, 
        vectors: torch.Tensor, 
        labels: torch.Tensor, 
        func: Callable, 
        depth: int
    ) -> DecisionNode:
        if len(vectors) == 0:
            return DecisionNode()
        if depth == 0:
            return DecisionNode(
                results=labels.mean().item()
            )
        current_score = func(labels)
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(vectors[0])
        for col in range(0, column_count):
            # @수정
            column_values = vectors[:, col].unique()
            for colval in column_values:
                vectors_set_1, values_set_1, vectors_set_2, values_set_2 = \
                    divide_set(vectors, labels, col, colval)

                p = float(len(vectors_set_1)) / len(vectors)
                gain = current_score - \
                    p * func(values_set_1) - \
                    (1 - p) * func(values_set_2)
                if (gain > best_gain and 
                    len(vectors_set_1) > 0 and 
                    len(vectors_set_2) > 0):
                    best_gain = gain
                    best_criteria = (col, colval)
                    best_sets = (
                        (vectors_set_1, values_set_1), 
                        (vectors_set_2, values_set_2)
                    )
        if best_gain > 0:
            # Recursive
            true_branch = self._build_tree(
                best_sets[0][0], 
                best_sets[0][1], 
                func, 
                depth-1
            )
            false_branch = self._build_tree(
                best_sets[1][0], 
                best_sets[1][1], 
                func, 
                depth-1
            )
            return DecisionNode(col=best_criteria[0],
                                value=best_criteria[1],
                                tb=true_branch, fb=false_branch)
        else:
            return DecisionNode(results=labels.mean().item())

    def predict(
        self, 
        vector: torch.Tensor
    ) -> Union[int, float]: # DecisionNode.results
        return self._regress(vector, self._root_node)

    def _regress(
        self, 
        vector: torch.Tensor, 
        node: DecisionNode,
    ) -> Union[int, float]: # DecisionNode.results
        if node.results is not None:
            return node.results
        else:
            if vector[:, node.col] >= node.value:
                branch = node.tb
            else:
                branch = node.fb

            return self._regress(vector, branch)
            
            
# Variance Function
variance = lambda v: v.var()

def variance(v):
    if v.dim() == 0:
        return 0
    else:
        return v.var()
        
        
# Random Forest Regressor
class TorchRandomForestRegressor(nn.Module):
    
    def __init__(
        self,  
        nb_trees: int, 
        nb_samples: int, 
        max_depth: int=-1, 
        bootstrap: bool=True
    ):
        super().__init__()
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def fit(
        self, 
        vectors: torch.Tensor, 
        labels: torch.Tensor
    ):
        for i in range(self.nb_trees):
            print(f"{i+1}번째 나무 실행 중")
            tree = TorchDecisionTreeRegressor(self.max_depth)
            list_features = sample_dimensions(vectors)
            self.trees_features.append(list_features)
            if self.bootstrap:
                # 어떤 행을 고를 것인가?
                sampled_vectors, sample_labels = sample_vectors(
                    vectors, labels, self.nb_samples)
                # @수정, 어떤 열을 고를 것인가?
                sampled_featured_vectors = sampled_vectors.index_select(
                    1, list_features)
                tree.fit(sampled_featured_vectors, sample_labels)
            else:
                # bootstrap안하고 열만 선택
                sampled_featured_vectors = vectors.index_select(
                    1, list_features)
                tree.fit(sampled_featured_vectors, labels)
            self.trees.append(tree)
            
    def _predict(
        self,
        vector: torch.Tensor,
    ) -> Union[int, float]:
        predictions_sum = 0
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = vector.index_select(0, index_features)
            predictions_sum += tree.predict(sampled_vector)
        return predictions_sum / len(self.trees)
        

    def predict(
        self, 
        vectors: torch.Tensor
    ) -> torch.Tensor:
        """ TODO: 병렬처리! """
        results = []
        for vector in vectors:
            results.append(self._predict(vector))
        return torch.Tensor(results)
