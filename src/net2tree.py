from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn


# -------------------------
# Tree objects
# -------------------------

@dataclass
class LeafRegressor:
    theta: np.ndarray  # (in_dim,)
    bias: float

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.theta, x) + self.bias)


@dataclass
class TreeNode:
    # Internal split
    A: Optional[np.ndarray] = None  # (in_dim,)
    b: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    # Leaf
    reg: Optional[LeafRegressor] = None

    def is_leaf(self) -> bool:
        return self.reg is not None

    def predict(self, x: np.ndarray) -> float:
        if self.reg is not None:
            return self.reg.predict(x)

        if self.A is None or self.b is None or self.left is None or self.right is None:
            raise RuntimeError("Malformed internal node.")

        score = float(np.dot(self.A, x) + self.b)

        # ---- CRITICAL: YOUR CONVENTION ----
        # D>0 corresponds to "True" condition list = left_parents => GO LEFT.
        return self.left.predict(x) if score > 0.0 else self.right.predict(x)


class ObliqueRegressionTree:
    def __init__(self, root: TreeNode, height: int, in_dim: int):
        self.root = root
        self.height = height
        self.in_dim = in_dim

    def predict_one(self, x: Union[np.ndarray, torch.Tensor, List[float]]) -> float:
        x_np = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_np.shape[0] != self.in_dim:
            raise ValueError(f"Expected ({self.in_dim},) got {x_np.shape}")
        return self.root.predict(x_np)

    def predict(self, X: Union[np.ndarray, torch.Tensor, List[List[float]]]) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim != 2 or X_np.shape[1] != self.in_dim:
            raise ValueError(f"Expected (N,{self.in_dim}) got {X_np.shape}")
        return np.array([self.root.predict(row) for row in X_np], dtype=np.float64)

    def get_agg_internal_coeffs(self) -> np.ndarray:
        """
        Get the aggregate internal split coefficients as a (in_dim,)
        by averaging the absolute values of the A vectors across all internal nodes.
        """
        A_list = []

        def traverse(node: TreeNode):
            if node.is_leaf():
                return
            A_list.append(node.A)
            traverse(node.left)
            traverse(node.right)

        traverse(self.root)

        if not A_list:
            raise ValueError("No internal nodes found in the tree.")

        A_stack = np.stack(A_list, axis=0)  # (num_internal_nodes, in_dim)
        agg_coeffs = np.mean(np.abs(A_stack), axis=0)  # (in_dim,)
        return agg_coeffs

    def get_agg_leave_coeffs(self) -> np.ndarray:
        """
        Get the aggregate leaf regressor coefficients as a (in_dim,)
        by averaging the absolute values of the theta vectors across all leaves.
        """
        theta_list = []

        def traverse(node: TreeNode):
            if node.is_leaf():
                theta_list.append(node.reg.theta)
                return
            traverse(node.left)
            traverse(node.right)

        traverse(self.root)

        if not theta_list:
            raise ValueError("No leaves found in the tree.")

        theta_stack = np.stack(theta_list, axis=0)  # (num_leaves, in_dim)
        agg_coeffs = np.mean(np.abs(theta_stack), axis=0)  # (in_dim,)
        return agg_coeffs


# -------------------------
# Parameter extraction helpers
# -------------------------

def _collapse_sequential_linears(seq: nn.Sequential) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collapse a Sequential of Linear layers WITHOUT nonlinearities into one affine map:
      Weff x + beff
    """
    linears = [m for m in seq.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("No Linear layers found in Sequential.")

    W_eff = linears[0].weight.detach().clone()
    b_eff = linears[0].bias.detach().clone()

    for lin in linears[1:]:
        W = lin.weight.detach().clone()
        b = lin.bias.detach().clone()
        b_eff = W @ b_eff + b
        W_eff = W @ W_eff

    return W_eff, b_eff


def _extract_internal_splits(model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      A: (k, in_dim)
      b: (k,)
    """
    linear1 = getattr(model, "linear1")

    if isinstance(linear1, nn.Linear):
        A = linear1.weight.detach().cpu().numpy()
        b = linear1.bias.detach().cpu().numpy()
        return A, b

    if isinstance(linear1, nn.Sequential):
        W, bb = _collapse_sequential_linears(linear1)
        return W.cpu().numpy(), bb.cpu().numpy()

    raise TypeError(f"Unsupported model.linear1 type: {type(linear1)}")


def _extract_leaf_regressors(model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """
    For linear_control=True, reg_hidden==0:
      regression_layer is nn.Linear(in_dim, num_leaves)
    Returns:
      theta: (L, in_dim)
      alpha: (L,)
    """
    reg = getattr(model, "regression_layer")
    if not isinstance(reg, nn.Linear):
        raise TypeError("Expected regression_layer to be nn.Linear(in_dim, num_leaves).")

    theta = reg.weight.detach().cpu().numpy()  # (L, in_dim)
    alpha = reg.bias.detach().cpu().numpy()    # (L,)
    return theta, alpha


# -------------------------
# Main converter
# -------------------------

def dtsemnet_to_tree(model: nn.Module) -> ObliqueRegressionTree:
    """
    Convert DTSemNet -> ObliqueRegressionTree consistent with your genDT() convention.
    DTSemNet must have been trained with linear_control=True and reg_hidden==0, so that the internal splits are linear and the leaf regressors are linear. 
    """
    if not bool(getattr(model, "is_regression")):
        raise ValueError("Expected is_regression=True.")
    if not bool(getattr(model, "linear_control")):
        raise ValueError("Expected linear_control=True for this exporter.")

    height = int(getattr(model, "_height"))
    in_dim = int(getattr(model, "input_dim"))
    k = int(getattr(model, "out_features"))
    L = int(getattr(model, "num_leaves"))

    expected_k = 2 ** height - 1
    expected_L = 2 ** height
    if k != expected_k or L != expected_L:
        raise ValueError(
            f"Complete-tree mismatch: got k={k},L={L} but height={height} "
            f"implies k={expected_k},L={expected_L}."
        )

    A_mat, b_vec = _extract_internal_splits(model)   # (k,in_dim), (k,)
    theta_mat, alpha_vec = _extract_leaf_regressors(model)  # (L,in_dim), (L,)

    # Build complete binary tree using heap indexing:
    # internal nodes: heap_idx 0..k-1
    # leaves: heap_idx k..k+L-1 ; leaf_id = heap_idx - k
    def build(heap_idx: int) -> TreeNode:
        if heap_idx >= k:
            leaf_id = heap_idx - k
            reg = LeafRegressor(theta=theta_mat[leaf_id].astype(np.float64),
                                bias=float(alpha_vec[leaf_id]))
            return TreeNode(reg=reg)

        left_idx = 2 * heap_idx + 1
        right_idx = 2 * heap_idx + 2

        return TreeNode(
            A=A_mat[heap_idx].astype(np.float64),
            b=float(b_vec[heap_idx]),
            left=build(left_idx),
            right=build(right_idx),
            reg=None,
        )

    root = build(0)
    return ObliqueRegressionTree(root=root, height=height, in_dim=in_dim)


# -------------------------
# Optional: quick consistency test
# -------------------------

@torch.no_grad()
def compare_tree_vs_model(model: nn.Module, tree: ObliqueRegressionTree,
                          X: Union[np.ndarray, torch.Tensor]) -> Tuple[float, float]:
    """
    Compare outputs on X. This checks only numerical equality of outputs,
    not routing equality.
    """
    model.eval()
    X_np = np.asarray(X, dtype=np.float32)
    X_t = torch.as_tensor(X_np, dtype=torch.float32)

    y_model = model(X_t).detach().cpu().numpy().reshape(-1)
    y_tree = tree.predict(X_np).reshape(-1)

    abs_err = np.abs(y_model - y_tree)
    return float(abs_err.max()), float(abs_err.mean())


if __name__ == "__main__":
    # Quick test: Create a model, convert to tree, and compare predictions.

    from dtsemnet import DTSemNet

    obs_dim = 1000
    feature_dim = 10
    depth = 3
    n_classes = 1
    model = DTSemNet(
        in_dim=feature_dim,
        out_dim=n_classes,
        height=depth,
        is_regression=True,
        over_param=[],
        linear_control=True,
        reg_hidden=0,
        wt_init=False,
    )
    tree = dtsemnet_to_tree(model)
    
    dummy_data = torch.randn(obs_dim, feature_dim)
    abs_error, mean_error = compare_tree_vs_model(model, tree, dummy_data)
    print(f"Max abs error: {abs_error:.6f}, Mean abs error: {mean_error:.6f}")