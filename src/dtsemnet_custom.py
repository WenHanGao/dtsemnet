import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LeafNode:
    left_parents: List[int]
    right_parents: List[int]
    leaf_act: List[int]
    

@dataclass
class DTParams:
    dim_in: int
    dim_out: int
    W: np.ndarray
    B: np.ndarray
    init_leaves: List[LeafNode]


def generate_complete_binary_tree(num_leaf: int, dim_out: int) -> List[LeafNode]:
    '''
    Generates a complete binary tree with the specified number of leaf nodes and output dimension.
        - num_leaf: The number of leaf nodes in the complete binary tree.
        - dim_out: The output dimension of the leaf nodes (classification: number of classes, regression: number of leave nodes).
    Returns a list of leaf nodes, where each leaf node is represented as a list containing:
        - left_parents: A list of indices of the parent nodes where the path to the leaf node takes the left branch.
        - right_parents: A list of indices of the parent nodes where the path to the leaf node takes the right branch.
        - leaf_act: A one-hot encoded list representing the action associated with the leaf node where the index of the action is set to 1 and the rest are set to 0.
    '''
    height = math.ceil(math.log2(num_leaf))
    leaf_nodes_lists = []

    controller_num = 0
    leaf_action = [0] * dim_out

    # Stack for DFS traversal of the complete binary tree. Each element in the stack is a tuple containing:
    # (current_node_index, left_parents_list, right_parents_list)
    stack = [(0, [], [])]

    while stack:
        node, left_parents, right_parents = stack.pop()

        left_child = 2 * node + 1
        right_child = 2 * node + 2

        if len(left_parents) + len(right_parents) >= height:  # Leaf node
            leaf_act = leaf_action.copy()
            leaf_act[controller_num] = 1
            controller_num += 1
            if controller_num == dim_out:
                controller_num = 0
            leaf_nodes_lists.append(LeafNode(left_parents, right_parents, leaf_act))
        else:
            stack.append(
                (right_child, left_parents.copy(), right_parents + [node]))
            stack.append(
                (left_child, left_parents + [node], right_parents.copy()))

    assert len(leaf_nodes_lists) == num_leaf, 'The number of leaf nodes is not correct'
    return leaf_nodes_lists


def generate_complete_binary_tree_custom(num_leaf: int, dim_out: int, custom_leaf_actions: List[int]) -> List[LeafNode]:
    '''
    Generates a complete binary tree with the specified number of leaf nodes, output dimension, and custom leaf actions.
        - num_leaf: The number of leaf nodes in the complete binary tree.
        - dim_out: The output dimension of the leaf nodes (classification: number of classes, regression: number of leave nodes).
        - custom_leaf_actions: A list of integers representing the action index for each leaf node. The length of this list should be equal to num_leaf and each integer should be in the range [0, dim_out-1].
    Returns a list of leaf nodes, where each leaf node is represented as a list containing:
        - left_parents: A list of indices of the parent nodes where the path to the leaf node takes the left branch.
        - right_parents: A list of indices of the parent nodes where the path to the leaf node takes the right branch.
        - leaf_act: A one-hot encoded list representing the action associated with the leaf node where the index of the action is set to 1 and the rest are set to 0, based on the custom_leaf_actions provided.
    '''
    leaf_action = [0] * dim_out
    height = math.ceil(math.log2(num_leaf))
    
    leaf_nodes_lists = []
    stack = [(0, [], [])]

    controller_num = 0

    while stack:
        node, left_parents, right_parents = stack.pop()

        left_child = 2 * node + 1
        right_child = 2 * node + 2

        if len(left_parents) + len(right_parents) >= height:  # Leaf node
            leaf_act = leaf_action.copy()
            leaf_act[custom_leaf_actions[controller_num]] = 1
            controller_num += 1
            leaf_nodes_lists.append(LeafNode(left_parents, right_parents, leaf_act))
        else:
            stack.append(
                (right_child, left_parents.copy(), right_parents + [node]))
            stack.append(
                (left_child, left_parents + [node], right_parents.copy()))

    assert len(leaf_nodes_lists) == num_leaf, 'The number of leaf nodes is not correct'

    return leaf_nodes_lists


def genDT(dim_in: int=784, dim_out: int=10, num_leaf: int=8, custom_leaf: Optional[List[int]]=None) -> DTParams:
    """
    Generates a decision tree with the specified parameters.
    W, B are initialized weights for the internal nodes of the decision tree
    
    Args:
    - dim_in: Input dimension (number of features)
    - dim_out: Output dimension (number of classes for classification or number of leaf nodes for regression)
    - num_leaf: Number of leaf nodes in the decision tree
    - custom_leaf: A list of integers representing the action index for each leaf node. The length of this list should be equal to num_leaf and each integer should be in the range [0, dim_out-1]. If None, a complete binary tree with default leaf actions will be generated

    Returns:
        DTParams: An object containing the generated decision tree parameters
    """
    num_nodes = num_leaf - 1

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    # W = np.random.randn(num_nodes, dim_in)  # each row represents a node
    # B = np.random.randn(num_nodes)  # Biases of each node

    W = np.random.randn(num_nodes, dim_in)  # each row represents a node
    B = np.random.randn(num_nodes)  # Biases of each node

    if custom_leaf:
        assert len(custom_leaf) == num_leaf, 'The number of leaf nodes is not correct'
        init_leaves = generate_complete_binary_tree_custom(num_leaf=num_leaf, dim_out=dim_out, custom_leaf_actions=custom_leaf)
        
    else:
        init_leaves = generate_complete_binary_tree(num_leaf=num_leaf, dim_out=dim_out)

    dt_params = DTParams(dim_in=dim_in, dim_out=dim_out, W=W, B=B, init_leaves=init_leaves)
    return dt_params


class MaxPoolLayer(nn.Module):
    """Custom MaxPool Layer for DTNet."""
    def __init__(self, leaf_actions):
        super(MaxPoolLayer, self).__init__()
        leaf_actions = np.array(leaf_actions, dtype='object')
        actions = np.unique(leaf_actions)
        self.node_groups = []
        for action in actions:
            self.node_groups.append(np.where(leaf_actions == action)[0])

        self.num_groups = len(self.node_groups)  #first 2 conditions

    def forward(self, x):
        batch_size, _ = x.size()
        out = torch.zeros((batch_size, self.num_groups), dtype=x.dtype, device=x.device)

        # loop over node groups and compute the maximum value for each group
        for i, group in enumerate(self.node_groups):
            out[:, i], _ = torch.max(x[:, group], dim=1)
            
        return out


class Sparser1(torch.autograd.Function):
    '''
    Straight-Through Estimator for the argmax function to get the leaf node for regression.
    In the forward pass, it outputs a one-hot encoded vector where the index of the maximum value in the input is set to 1 and the rest are set to 0.
    In the backward pass, it passes the gradient through unchanged (identity function) to allow for gradient flow through the argmax operation.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        inpargmax = input.argmax(-1)
        output = torch.zeros_like(input)
        output[torch.arange(input.shape[0]), inpargmax] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


class DTSemNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        is_regression: bool,
        height: int,
        over_param: list,
        reg_hidden: int,
        linear_control: bool = True,
        wt_init: bool = False,
        custom_leaf = None,
    ):
        '''
        DTNet is a neural network architecture that represent an oblique decision tree as a neural network. 
        Args:
            - in_dim : Input dimension (number of features)
            - out_dim : Output dimension (number of classes for classification or number of leaf nodes for regression)
            - is_regression : Boolean flag indicating whether the task is regression (True) or classification (False)
            - height : Height of the decision tree (number of layers in the tree)
            - over_param : List of scale factors for overparameterization. If empty, no overparameterization is applied. 
                            If not empty, it should contain scale factors for each layer of the overparameterization.
                            For example, if height=3 and over_param=[2, 4], it means that between the input and the internal node layer
                            there will be 2 hidden layers of overparameterization by internal nodes with scale factors 2 and 4 respectively.
            - linear_control : Boolean flag indicating whether to use a linear layer for regression control (True) or not (False). 
                            If True, an additional linear layer will be added to the regression branch of the network to allow for more flexible regression outputs.
            - wt_init : Boolean flag indicating whether to initialize the weights of the first linear layer with the weights of the decision tree (True) or to use random initialization (False). 
            - reg_hidden : Integer specifying the number of hidden units in the regression control layer if linear_control is True. If 0, no hidden layer will be used and the regression control will be a direct linear mapping from input to output.
            - custom_leaf : Optional list of integers representing the action index for each leaf node.
        '''
        super().__init__()

        
        self._height = height
        self._over_param = over_param
        self.is_regression = is_regression
        self.linear_control = linear_control

        int_nodes = 2 ** height - 1
        leaf_nodes = 2 ** height

        if not is_regression: # classification
            assert out_dim > 1, "out_dim must be greater than 1 for classification"
            t_out_dim = out_dim
        else: # regression: out_dim is the number of leaf nodes for regression
            t_out_dim = leaf_nodes
            
        # Get DT and required parameters
        if custom_leaf:
            self.custom_leaf_unique = list(set(custom_leaf))
        else:
            self.custom_leaf_unique = None
        
        dt_params = genDT(dim_in=in_dim, dim_out=t_out_dim,num_leaf=leaf_nodes, custom_leaf=custom_leaf)
        self.L, self.A = self.leaves_to_weight(dt_params.init_leaves, dt_params.W.shape[0])

        self.input_dim = in_dim  # input dimension of state 
        self.output_dim = out_dim  # output dimension of action
        self.out_features = len(dt_params.B)  # number of nodes in layer1 = number of nodes in DT
        self.num_leaves = len(self.A)  # number of leaves in DT
        self.init_weights = torch.tensor(dt_params.W, dtype=torch.float32)
        self.init_biases = torch.tensor(dt_params.B, dtype=torch.float32)

        # Layer1: Linear Layer
        # Overparams go here
        if len(self._over_param) == 0:
            self.linear1 = nn.Linear(self.input_dim, self.out_features)
            
            # Random initialization of weights (Orthogonal Initialization)
            if wt_init:
                with torch.no_grad(): nn.init.zeros_(self.linear1.bias)
                self.linear1.weight.data = self.init_weights  # intialize weights from DT weights
                nn.init.orthogonal_(self.linear1.weight)
                
            self.linear1 = nn.Sequential(self.linear1) 
            
        else:
            self.linear1 = []
            a = in_dim
            for scale_factor in self._over_param:
                b = int(scale_factor * int_nodes)
                self.linear1.append(nn.Linear(a, b))
                if self._over_param_nonlin:
                    self.linear1.append(nn.ReLU())
                self.linear1.append(nn.Dropout(0.1))
                a = b
                
            self.linear1.append(nn.Linear(a, int_nodes))
            
            with torch.no_grad():
                [nn.init.zeros_(x.bias) for x in self.linear1 if isinstance(x, nn.Linear)]
            self.linear1 = nn.Sequential(*self.linear1)
            
            # orthogonal initialization of linear1
            if wt_init:
                for m in self.linear1.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight)
        

        # Layer2: Relu Activation for Relu(+D) and Relu(-D)
        self.reluP = nn.ReLU()
        self.reluM = nn.ReLU()

        # Layer3: Linear Layer (w/o) activation to add various conditions
        # Weights linking from +D and -D to the leaf nodes are fixed {0, 1}
        self.linear2 = nn.Linear(
            2 * self.out_features,
            self.num_leaves,
            bias=False
        )
        self.init_weights2 = torch.tensor(self.L, dtype=torch.float32)
        self.linear2.weight.data = self.init_weights2  # intialize weights from DT
        self.linear2.weight.requires_grad = False  # Fixed weights (no gradient)


        # Layer4: MaxPool Layer
        self.leaf_actions = self.A
        
        
        # regression layer:
        if self.is_regression:
            # Layer5: Softmax (Applied in 'forward' method)
            self.softmax = nn.Softmax(dim=-1)

            if self.linear_control:
                if reg_hidden > 0:
                    reg_layers = [
                        nn.Linear(self.input_dim, int(reg_hidden)),
                        nn.Linear(int(reg_hidden), t_out_dim)
                    ]
                    self.regression_layer = nn.Sequential(*reg_layers)
                    
                else:
                    self.regression_layer = nn.Linear(self.input_dim, t_out_dim, bias=True)

            else:
                # constant regression output for each leaf node (no control from input features)
                self.regression_layer = nn.Linear(t_out_dim, 1, bias=True)
        else:
            self.mpool = MaxPoolLayer(self.leaf_actions)
            
            

    def forward(
        self,
        in_x: torch.Tensor,
    ) -> torch.Tensor: # type: ignore

        # Flatten the input if it is not already in the shape (batch_size, input_dim)
        in_x = in_x.view(in_x.size(0), -1)       
        
        x = self.linear1(in_x)  # shape (batch_size, num_int_nodes)
        if self.is_regression:
            if self.linear_control:
                regression_out = self.regression_layer(in_x)    # shape (batch_size, t_out_dim)
            else:
                regression_fn = self.regression_layer
        

        # Layer 2 Activation Layer (Relu(+D) and Relu(-D))
        relu_x = self.reluP(x)
        relu_neg_x = self.reluM(-x)
        x = torch.cat((relu_x, relu_neg_x), dim=1)  # shape (batch_size, 2 * num_int_nodes)

        # Layer 3 Linear Layer (w/o) activation to add various conditions (node of each leaves)
        x = self.linear2(x)  # shape (batch_size, num_leaves)

        if self.is_regression: # for regression

            # STE approximation of argmax to get the leaf node for regression
            x = self.softmax(x)
            x_hard = Sparser1.apply(x)

            # Allows x to be one-hot encoded while keeping gradients flowing through the softmax output during backpropagation
            x = x_hard - 0.5*x.detach() + 0.5*x 
            
            if self.linear_control:
                # multiply the regression output with the leaf node output to get the final output
                x = x * regression_out
                x = x.sum(dim=-1).reshape(-1, 1)
            else:
                x = regression_fn(x)
            
            return x
        
        else: # for classification
            # Layer 4 MaxPool Layer to get the max value for each leaf node
            x = self.mpool(x)
            if self.custom_leaf_unique:
                batch_size, original_dim = x.size()
                expanded_dim = self.output_dim
                #
                expanded_output = torch.zeros(batch_size, expanded_dim, dtype=x.dtype, device=x.device)

                # Assign the original values to the specified indices
                expanded_output[:, self.custom_leaf_unique[:original_dim]] = x
                expanded_output = torch.softmax(expanded_output, dim=-1)
                return expanded_output
            
            return x
    
    def leaves_to_weight(self, init_leaves: List[LeafNode], num_nodes: int):
        """
        Converts the leaves of the decision tree to the weights of the DTNet.
        Args:
            - init_leaves: List of leaf nodes, where each leaf node is represented as a list containing:
                - left_parents: A list of indices of the parent nodes where the path to the leaf node takes the left branch.
                - right_parents: A list of indices of the parent nodes where the path to the leaf node takes the right branch.
                - leaf_act: A one-hot encoded list representing the action associated with the leaf node where the index of the action is set to 1 and the rest are set to 0.
            - num_nodes: The number of internal nodes in the decision tree (which is equal to the number of nodes in the first linear layer of the DTNet).
        Returns:
            - L: Shape (num_leaves, 2 * num_nodes) matrix of weights of condition layer in DTNet (layer 3)
            - A: Shape (num_leaves,) array of actions associated with each leaf node
        """
        num_leaves = len(init_leaves)
        L = np.zeros((num_leaves, 2 * num_nodes))  # each row represents a leaf
        A = np.zeros(num_leaves)  # action associated with each leaf
        
        for ix, leaf in enumerate(init_leaves):
            if len(leaf.left_parents) > 0:  # True Conditions
                # For Relu(D) for each node
                L[ix][leaf.left_parents] = 1
                
            if len(leaf.right_parents) > 0:  # False Conditions
                # For Relu(-D) for each node
                L[ix][[nx + num_nodes for nx in leaf.right_parents]] = 1

            # don't care conditions: need to but D+ and D- for each node
            dont_cares = set(range(num_nodes)) - set(leaf.left_parents + leaf.right_parents)  # don't care conditions
            if len(dont_cares) > 0:
                L[ix][list(dont_cares)] = 1
                L[ix][[nx + num_nodes for nx in dont_cares]] = 1

            A[ix] = np.argmax(leaf.leaf_act)  # action associated with leaf 0: Left and 1: Right

        return L, A
