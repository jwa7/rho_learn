In this directory are a linear and nonlinear model pretrained on the dataset of
1000 water monomers, $N_{\text{train}} = 800, N_{\text{test}} = 199$. For both,
LBFGS is used as the optimizer with a learning rate of 1.25. MSE is used as the
loss, and invariants are standardized prior to training. The averages of
invariant features of the training data are also provided. The test error of both
models saturated by epoch 2500.

The nonlinear model uses the architecture described in the figure at
[docs/example/figures/nonlinear_architecture.png](https://github.com/m-stack-org/rho_learn/blob/main/docs/example/figures/nonlinear_architecture.png).
The neural network applied to invariant blocks features a 2 hidden layers, each
consisting of a linear layer with a width of 16 nodes, and a SiLU activation
function.


To load and make predictions with the models:

```py
import os
import torch
import equistore
from rholearn import io, utils

RHOLEARN_PATH = "/path/to/rho_learn/"

# Load X data and build representation
# input = ...

model = io.load_torch_object(
    path=os.path.join(
        RHOLEARN_PATH, 
        "docs/example/water/runs/pretrained_models/", 
        "nonlinear.pt"
    ),
    device=torch.device('cpu'),
    torch_obj_str="model",
)

# Make a prediction, with no gradient tracking 
with torch.no_grad():
    out_pred = model(input)

# Un-standardize the invariants
inv_means = equistore.load(
    os.path.join(
        RHOLEARN_PATH, 
        "docs/example/water/runs/pretrained_models/", 
        "inv_means.npz"
    )
)
out_pred = utils.standardize_invariants(
    tensor=utils.tensor_to_numpy(out_pred),  # need to convert tensor to numpy
    invariant_means=inv_means,
    reverse=True,  # i.e. un-standardize
)
```