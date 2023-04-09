# TN_CFD
CFD simulations using Tensor Network encoding.
Starting point https://www.nature.com/articles/s43588-021-00181-1

## Folder 2D_TDJ 
In "direct_numerical_simulation.ipynb" the DNS scheme for the "temporally developing jet" problem is demonstrated. The method itself is explained in https://doi.org/10.1016/j.cpc.2016.02.023
Next steps: Calculate pressure from velocity field or use different projection method? Define new problem with no-slip boundary conditions (turbulent channel flow?). How do we encode boundaries of objects in the finite difference method? Should we use finite volumes instead? 

In "MPS_simulation.ipynb" the encoding of the velocityfield into the MPS representation is outlined. The scale resolving encoding is expalined in the original paper. Differentiation operators are not defined yet. 
Next steps: Try tensor networks library (quimb or cutensor?).

## Folder miscellaneous
Old code versions which could help in the future.