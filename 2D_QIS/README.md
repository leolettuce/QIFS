# Quantum-Inspired Simulations
This folder contains three different implementations of the quantum-inspired CFD algorithm.

## cuTensorNet
This implementation is utilizing NVIDIA's cuTensorNet library (part of cuQuantum).
As the tensor operations are done on (NVIDIA) GPUs, it is the fastest implementation.
Please also install cuQuantum: https://docs.nvidia.com/cuda/cuquantum/latest/python/

## quimb
This implementation is utilizing the tensor network library 'quimb'.
As it is running on CPUs, only an installation of quimb is required: https://quimb.readthedocs.io/en/latest/

## numpy
This implementation is utilizing the basic functions of numpy.
It is incredibly slow.

## Other files
The file 'convert.py' converts tensor networks between cuTensorNet and quimb formats.
It is only used for debugging purposes.
