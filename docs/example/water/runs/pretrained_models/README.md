In this directory are a linear (`linear.pt`) and nonlinear (`nonlinear.pt`)
model pretrained on the dataset of 1000 water monomers
(`water/data/water_monomers_1k.xyz`), with $N_{\text{train}} = 800,
N_{\text{test}} = 199$. For both models, LBFGS is used as the optimizer with a
learning rate of 1.25. Invariant blocks of the electron density are standardized
prior to training, with the averages of the invariant features of the training
set provided (`water/data/inv_means.npz`). The test error of both models
saturated by epoch 2500.

Purely for demonstrative purposes, and to keep the code lightweight, an MSE loss
metric on the basis set coefficients is used to train the model. In practice,
the proper way of defining a loss would be to take the product of a) the
difference in predicted and target coefficients, $\Delta c$, and b) the
difference in predicted and target projections of the coefficients onto the
basis set, $\Delta w$. This requires storage of and operations with large
overlap matrices at runtime, with is costly and therefore not included in this
water training demo. This approach is included in the azoswitch example,
however, where Coulomb metrics (instead of overlap matrices) are used to define
the loss.

The nonlinear model uses the architecture described in the figure at
[docs/example/figures/nonlinear_architecture.png](https://github.com/m-stack-org/rho_learn/blob/main/docs/example/figures/nonlinear_architecture.png).
The neural network applied to invariant blocks features a 2 hidden layers, each
consisting of a linear layer with a width of 16 nodes, and a SiLU activation
function.


To load and make predictions with the models:

```py
import os
import torch
import metatensor
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
inv_means = metatensor.load(
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