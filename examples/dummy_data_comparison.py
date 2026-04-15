'''
This script demonstrates the training of a DTSemNet on a synthetic dataset generated from a simple oblique tree data generating process. 
The performance of the trained DTSemNet is compared against an XGBoost model trained on the same dataset.
The synthetic dataset is created by defining an oracle function that simulates a decision tree with oblique splits.
The output scatter plot is saved to the 'data/figures' directory, showing the true values against the predictions from both models for the validation set.
'''


from pathlib import Path
import sys
    
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
    
# from src.dtsemnet_custom import DTSemNet
from src.dtsemnet import DTSemNet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb


if __name__ == "__main__":    
    torch.manual_seed(11)
    np.random.seed(11)
    torch.cuda.manual_seed(11)

    # Example of 2 level tree for comparison with XGBoost

    obs_dim = 1000
    feature_dim = 10
    depth = 2

    int_node_weights = np.random.rand(2**depth - 1, feature_dim)  # weights for internal nodes of the decision tree
    leave_node_weights = np.random.rand(2**depth, feature_dim)  # weights for leaf nodes of the decision tree

    def oracle_fn(x: np.ndarray) -> float:
        condition1 = np.dot(int_node_weights[0], x) > 0  # condition for root node
        condition2 = np.dot(int_node_weights[1], x) > 0  # condition for left child of root node
        condition3 = np.dot(int_node_weights[2], x) > 0  # condition for right child of root node

        if condition1 and condition2:
            return np.dot(leave_node_weights[0], x)
        elif condition1 and not condition2:
            return np.dot(leave_node_weights[1], x)
        elif not condition1 and condition3:
            return np.dot(leave_node_weights[2], x)
        else:
            return np.dot(leave_node_weights[3], x)
    

    x = torch.randn(obs_dim, feature_dim)
    y = torch.tensor([oracle_fn(xi.numpy()) for xi in x], dtype=torch.float32).view(-1, 1)
    y += 0.01 * torch.randn_like(y)  # add some noise to the target values
    
    feature_names = [f"feature_{i}" for i in range(x.size(1))] 

    in_dim = len(feature_names)
    n_classes = 1
    n_leaves = 2 ** depth
    model = DTSemNet(
        in_dim=in_dim,
        out_dim=n_classes,
        height=depth,
        is_regression=True,
        over_param=[],
        linear_control=True,
        reg_hidden=0,
        wt_init=False,
    )

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    tr_samples = 800
    x_train = x[:tr_samples]
    y_train = y[:tr_samples]
    x_val = x[tr_samples:]
    y_val = y[tr_samples:]
    n_epochs = 5000
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    for epoch in range(n_epochs):
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        total_loss /= len(train_loader)
        total_val_loss = 0.0
        for x_batch, y_batch in val_loader:
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_batch)
                val_loss = criterion(val_outputs, y_batch)
                total_val_loss += val_loss.item() * x_batch.size(0)
        total_val_loss /= len(val_loader)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {total_loss:.4f}, Val Loss: {total_val_loss:.4f}")

    model.eval()
    y_val_pred = model(x_val)
    y_val_pred_np = y_val_pred.detach().numpy().ravel()
    y_val_true_np = y_val.numpy().ravel()
    mse = np.mean((y_val_true_np - y_val_pred_np) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(y_val_true_np - y_val_pred_np) / np.abs(y_val_true_np))
    r2 = 1 - np.sum((y_val_true_np - y_val_pred_np) ** 2) / np.sum((y_val_true_np - np.mean(y_val_true_np)) ** 2)
    dt_title = f"DTNet Validation MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}"

    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(x_train.numpy(), y_train.numpy().ravel())
    xgb_val_pred = xgb_model.predict(x_val.numpy())
    mse = np.mean((y_val_true_np - xgb_val_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(y_val_true_np - xgb_val_pred) / np.abs(y_val_true_np))
    r2 = 1 - np.sum((y_val_true_np - xgb_val_pred) ** 2) / np.sum((y_val_true_np - np.mean(y_val_true_np)) ** 2)
    xgb_title = f"XGBoost Validation MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}"

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(y_val.numpy(), y_val_pred.detach().numpy(), label='DTNet Predictions', alpha=0.5)
    ax.scatter(y_val.numpy(), xgb_val_pred, label='XGBoost Predictions', alpha=0.5)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', label='Ideal Predictions')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'DTNet vs XGBoost Predictions\n{dt_title}\n{xgb_title}')
    ax.legend()
    
    output_dir = BASE_DIR / 'data' / 'figures' / 'dtsem_xgb_compare_dummy_true_vs_pred.png'
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir)
 