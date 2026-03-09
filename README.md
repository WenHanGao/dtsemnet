DTSemNet 
========

Implementation of the DTSemNet architecture, as proposed in the paper:
“Vanilla Gradient Descent for Oblique Decision Trees,” ECAI-2024.
[[Paper]](https://arxiv.org/pdf/2408.09135) [[Website]](https://cps-research-group.github.io/dtsemnet)

# Main Class
`src/dtsemnet_custom.py` contains the main model class `DTSemNet`. The script is based on the original script `src/dtsemnet.py` with some additional comments and dataclasses for understanding. Some unused functions and classes were also removed.

# Examples
`examples/dummy_data_comparison.py` contains a script that test the performance of oblique tree with XGBoost on a simple dummy data with true oblique-tree-data-generating process.

# Convert to Tree
`src/net2tree.py` contains the function `dtsemnet_to_tree` which converts the weights from the neural network into an oblique tree class (`ObliqueRegressionTree`). Note that there some some contraints for the conversion. (The conversion script is mainly generated using AI)
