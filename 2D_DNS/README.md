# Direct Numerical Simulation
This folder contains two different implementations of the Direct Numerical Simulation.
The implementation is based on https://doi.org/10.1016/j.cpc.2016.02.023

## DNS_cupy.py
This implementation is utilizing NVIDIA's cupy library.
As the tensor operations are done on (NVIDIA) GPUs, it is the fastest implementation.
Please also install cupy: https://cupy.dev/

## DNS_numpy.py
This implementation is utilizing the basic functions of numpy.

## direct_numerical_simulation.ipynb
This is a notebook containing the same code as above.
