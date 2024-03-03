#  Copyright 2024 Leonhard Hoelscher. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Differential operators in form of MPOs
import numpy as np
import cupy as cp
from cuquantum.cutensornet.tensor import decompose, SVDMethod
from cuquantum.cutensornet.experimental import contract_decompose
from cuquantum import contract, einsum, tensor


def Diff_1_2_y_MPO(n, dx, options=None):
    # first order derivative with second order precision in y direction
    # tensor stucture:
    #                          |
    # left_terminator - (central_node)^n - right_terminator
    #                          |
    left_terminator = cp.array([1, 1, 1], dtype='float64').reshape(1, 3)
    central_node = cp.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    central_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    central_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    central_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: subtract 1
    central_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for subtraction (2)
    central_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    central_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for subtraction (2)
    central_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry(0)
    right_terminator = 1/dx*cp.array([0, -1/2, 1/2], dtype='float64').reshape(3, 1) # coefficients

    # Define arrays for MPO
    left_node = contract('ab, burd->aurd', left_terminator, central_node, options=options)
    right_node = contract('lurd, re->lued', central_node, right_terminator, options=options)
    arrays = [left_node] + [central_node]*(n-2) + [right_node]

    return arrays


def Diff_1_2_x_MPO(n, dx, options=None):
    # first order derivative with second order precision in x direction
    # tensor stucture:
    #                          |
    # left_terminator - (central_node)^n - right_terminator
    #                          |
    left_terminator = cp.array([1, 1, 1], dtype='float64').reshape(1, 3)
    central_node = cp.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    central_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    central_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    central_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: subtract 1
    central_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for subtraction (2)
    central_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for subtraction (2)
    central_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    central_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    right_terminator = 1/dx*cp.array([0, -1/2, 1/2], dtype='float64').reshape(3, 1) # coefficients

    # Define arrays for MPO
    left_node = contract('ab, burd->aurd', left_terminator, central_node, options=options)
    right_node = contract('lurd, re->lued', central_node, right_terminator, options=options)
    arrays = [left_node] + [central_node]*(n-2) + [right_node]

    return arrays


def Diff_2_2_y_MPO(n, dx, options=None):
    # second order derivative with second order precision in y direction
    # tensor stucture:
    #                          |
    # left_terminator - (central_node)^n - right_terminator
    #                          |
    left_terminator = cp.array([1, 1, 1], dtype='float64').reshape(1, 3)
    central_node = cp.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    central_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    central_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    central_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: subtract 1
    central_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for subtraction (2)
    central_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    central_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for subtraction (2)
    central_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry(0)
    right_terminator = 1/dx**2*cp.array([-2, 1, 1], dtype='float64').reshape(3, 1) # coefficients

    # Define arrays for MPO
    left_node = contract('ab, burd->aurd', left_terminator, central_node, options=options)
    right_node = contract('lurd, re->lued', central_node, right_terminator, options=options)
    arrays = [left_node] + [central_node]*(n-2) + [right_node]

    return arrays


def Diff_2_2_x_MPO(n, dx, options=None):
    # second order derivative with second order precision in x direction
    # tensor stucture:
    #                          |
    # left_terminator - (central_node)^n - right_terminator
    #                          |
    left_terminator = cp.array([1, 1, 1], dtype='float64').reshape(1, 3)
    central_node = cp.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    central_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    central_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    central_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: subtract 1
    central_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for subtraction (2)
    central_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for subtraction (2)
    central_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    central_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    right_terminator = 1/dx**2*cp.array([-2, 1, 1], dtype='float64').reshape(3, 1) # coefficients

    # Define arrays for MPO
    left_node = contract('ab, burd->aurd', left_terminator, central_node, options=options)
    right_node = contract('lurd, re->lued', central_node, right_terminator, options=options)
    arrays = [left_node] + [central_node]*(n-2) + [right_node]

    return arrays


def Diff_1_8_y_MPO(n, dx, options=None):
    # first order derivative with eigth order precision in y direction
    # tensor stucture:
    #                          |                         |               |
    # left_terminator - (central_node)^(n-2) - second to last node - last node - right_terminator
    #                          |                         |               |
    # left terminator
    left_terminator = cp.array([1, 1, 1], dtype='float64').reshape(1, 3)
    # central node
    central_node = cp.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    central_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    central_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    central_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: subtract 1
    central_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for subtraction (2)
    central_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    central_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for subtraction (2)
    central_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry(0)
    # second to last node
    sec_last_node = cp.zeros((3, 4, 5, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    sec_last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    sec_last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    sec_last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    sec_last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    sec_last_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    sec_last_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    sec_last_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    sec_last_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: subtract 1
    sec_last_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for subtraction (2)
    sec_last_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    sec_last_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for subtraction (2)
    sec_last_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry (0)
    # 3 from right: identity with carry for addition
    sec_last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    sec_last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    sec_last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    sec_last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    # 4 from right: identity with carry for subtraction
    sec_last_node[2, 0, 4, 0] =  1   # 00 -> 00 carry for subtraction (2)
    sec_last_node[2, 1, 4, 1] =  1   # 01 -> 01 carry for subtraction (2)
    sec_last_node[2, 2, 4, 2] =  1   # 10 -> 10 carry for subtraction (2)
    sec_last_node[2, 3, 4, 3] =  1   # 11 -> 11 carry for subtraction (2)
    # last node
    last_node = cp.zeros((5, 4, 6, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    last_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    last_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    last_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    last_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: subtract 1
    last_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for subtraction (2)
    last_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    last_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for subtraction (2)
    last_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry (0)
    # 3 from right: add and subtract 2
    last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    last_node[2, 0, 3, 0] = -1   # 00 -> 00 carry for subtraction (2)
    last_node[2, 1, 3, 1] = -1   # 01 -> 01 carry for subtraction (2)
    last_node[2, 2, 3, 2] = -1   # 10 -> 10 carry for subtraction (2)
    last_node[2, 3, 3, 3] = -1   # 11 -> 11 carry for subtraction (2)
    # 4 from right: add and subtract 3
    last_node[1, 0, 4, 1] = 1    # 00 -> 01 carry for addition (1)
    last_node[3, 1, 4, 0] = 1    # 01 -> 00 carry for addition one further (3)
    last_node[1, 2, 4, 3] = 1    # 10 -> 11 carry for addition (1)
    last_node[3, 3, 4, 2] = 1    # 11 -> 10 carry for addition one further (3)
    last_node[4, 0, 4, 1] = -1   # 00 -> 01 carry for subtraction one further (4)
    last_node[2, 1, 4, 0] = -1   # 01 -> 00 carry for subtraction (2)
    last_node[4, 2, 4, 3] = -1   # 10 -> 11 carry for subtraction one further (4)
    last_node[2, 3, 4, 2] = -1   # 11 -> 10 carry for subtraction (2)
    # 5 from right: add and subtract 4
    last_node[3, 0, 5, 0] = 1    # 00 -> 00 carry for addition one further (3)
    last_node[3, 1, 5, 1] = 1    # 01 -> 01 carry for addition one further (3)
    last_node[3, 2, 5, 2] = 1    # 10 -> 10 carry for addition one further (3)
    last_node[3, 3, 5, 3] = 1    # 11 -> 11 carry for addition one further (3)
    last_node[4, 0, 5, 0] = -1   # 00 -> 00 carry for subtraction one further (4)
    last_node[4, 1, 5, 1] = -1   # 01 -> 01 carry for subtraction one further (4)
    last_node[4, 2, 5, 2] = -1   # 10 -> 10 carry for subtraction one further (4)
    last_node[4, 3, 5, 3] = -1   # 11 -> 11 carry for subtraction one further (4)
    # right terminator
    right_terminator = 1/dx*cp.array([0, -4/5, 4/5, 1/5, -4/105, 1/280], dtype='float64').reshape(6, 1) # coefficients

    # Define arrays for MPO
    left_node = contract('ab, burd->aurd', left_terminator, central_node, options=options)
    right_node = contract('lurd, re->lued', last_node, right_terminator, options=options)
    arrays = [left_node] + [central_node]*(n-3) + [sec_last_node] + [right_node]

    return arrays


def Diff_1_8_x_MPO(n, dx, options=None):
    # first order derivative with eighth order precision in x direction
    # tensor stucture:
    #                          |                         |               |
    # left_terminator - (central_node)^(n-2) - second to last node - last node - right_terminator
    #                          |                         |               |
    # left terminator
    left_terminator = cp.array([1, 1, 1], dtype='float64').reshape(1, 3)
    # central node
    central_node = cp.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    central_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    central_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    central_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: subtract 1
    central_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for subtraction (2)
    central_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for subtraction (2)
    central_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    central_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # second to last node
    sec_last_node = cp.zeros((3, 4, 5, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    sec_last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    sec_last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    sec_last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    sec_last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    sec_last_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    sec_last_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    sec_last_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    sec_last_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: subtract 1
    sec_last_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for subtraction (2)
    sec_last_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for subtraction (2)
    sec_last_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    sec_last_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # 3 from right: identity with carry for addition
    sec_last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    sec_last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    sec_last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    sec_last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    # 4 from right: identity with carry for subtraction
    sec_last_node[2, 0, 4, 0] =  1   # 00 -> 00 carry for subtraction (2)
    sec_last_node[2, 1, 4, 1] =  1   # 01 -> 01 carry for subtraction (2)
    sec_last_node[2, 2, 4, 2] =  1   # 10 -> 10 carry for subtraction (2)
    sec_last_node[2, 3, 4, 3] =  1   # 11 -> 11 carry for subtraction (2)
    # last node
    last_node = cp.zeros((5, 4, 6, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    last_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    last_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    last_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    last_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: subtract 1
    last_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for subtraction (2)
    last_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for subtraction (2)
    last_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    last_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # 3 from right: add and subtract 2
    last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    last_node[2, 0, 3, 0] = -1   # 00 -> 00 carry for subtraction (2)
    last_node[2, 1, 3, 1] = -1   # 01 -> 01 carry for subtraction (2)
    last_node[2, 2, 3, 2] = -1   # 10 -> 10 carry for subtraction (2)
    last_node[2, 3, 3, 3] = -1   # 11 -> 11 carry for subtraction (2)
    # 4 from right: add and subtract 3
    last_node[1, 0, 4, 2] = 1    # 00 -> 10 carry for addition (1)
    last_node[1, 1, 4, 3] = 1    # 01 -> 11 carry for addition (1)
    last_node[3, 2, 4, 0] = 1    # 10 -> 00 carry for addition one further (3)
    last_node[3, 3, 4, 1] = 1    # 11 -> 01 carry for addition one further (3)
    last_node[4, 0, 4, 2] = -1   # 00 -> 10 carry for subtraction one further (4)
    last_node[4, 1, 4, 3] = -1   # 01 -> 11 carry for subtraction one further (4)
    last_node[2, 2, 4, 0] = -1   # 10 -> 00 carry for subtraction (2)
    last_node[2, 3, 4, 1] = -1   # 11 -> 01 carry for subtraction (2)
    # 5 from right: add and subtract 4
    last_node[3, 0, 5, 0] = 1    # 00 -> 00 carry for addition one further (3)
    last_node[3, 1, 5, 1] = 1    # 01 -> 01 carry for addition one further (3)
    last_node[3, 2, 5, 2] = 1    # 10 -> 10 carry for addition one further (3)
    last_node[3, 3, 5, 3] = 1    # 11 -> 11 carry for addition one further (3)
    last_node[4, 0, 5, 0] = -1   # 00 -> 00 carry for subtraction one further (4)
    last_node[4, 1, 5, 1] = -1   # 01 -> 01 carry for subtraction one further (4)
    last_node[4, 2, 5, 2] = -1   # 10 -> 10 carry for subtraction one further (4)
    last_node[4, 3, 5, 3] = -1   # 11 -> 11 carry for subtraction one further (4)
    # right terminator
    right_terminator = 1/dx*cp.array([0, -4/5, 4/5, 1/5, -4/105, 1/280], dtype='float64').reshape(6, 1) # coefficients

    # Define arrays for MPO
    left_node = contract('ab, burd->aurd', left_terminator, central_node, options=options)
    right_node = contract('lurd, re->lued', last_node, right_terminator, options=options)
    arrays = [left_node] + [central_node]*(n-3) + [sec_last_node] + [right_node]

    return arrays


def Diff_2_8_y_MPO(n, dx, options=None):
    # second order derivative with eigth order precision in y direction
    # tensor stucture:
    #                          |                         |               |
    # left_terminator - (central_node)^(n-2) - second to last node - last node - right_terminator
    #                          |                         |               |
    # left terminator
    left_terminator = cp.array([1, 1, 1], dtype='float64').reshape(1, 3)
    # central node
    central_node = cp.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    central_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    central_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    central_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: subtract 1
    central_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for subtraction (2)
    central_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    central_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for subtraction (2)
    central_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry(0)
    # second to last node
    sec_last_node = cp.zeros((3, 4, 5, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    sec_last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    sec_last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    sec_last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    sec_last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    sec_last_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    sec_last_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    sec_last_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    sec_last_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: subtract 1
    sec_last_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for subtraction (2)
    sec_last_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    sec_last_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for subtraction (2)
    sec_last_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry (0)
    # 3 from right: identity with carry for addition
    sec_last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    sec_last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    sec_last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    sec_last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    # 4 from right: identity with carry for subtraction
    sec_last_node[2, 0, 4, 0] =  1   # 00 -> 00 carry for subtraction (2)
    sec_last_node[2, 1, 4, 1] =  1   # 01 -> 01 carry for subtraction (2)
    sec_last_node[2, 2, 4, 2] =  1   # 10 -> 10 carry for subtraction (2)
    sec_last_node[2, 3, 4, 3] =  1   # 11 -> 11 carry for subtraction (2)
    # last node
    last_node = cp.zeros((5, 4, 6, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    last_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    last_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    last_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    last_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: subtract 1
    last_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for subtraction (2)
    last_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    last_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for subtraction (2)
    last_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry (0)
    # 3 from right: add and subtract 2
    last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    last_node[2, 0, 3, 0] = 1    # 00 -> 00 carry for subtraction (2)
    last_node[2, 1, 3, 1] = 1    # 01 -> 01 carry for subtraction (2)
    last_node[2, 2, 3, 2] = 1    # 10 -> 10 carry for subtraction (2)
    last_node[2, 3, 3, 3] = 1    # 11 -> 11 carry for subtraction (2)
    # 4 from right: add and subtract 3
    last_node[1, 0, 4, 1] = 1    # 00 -> 01 carry for addition (1)
    last_node[3, 1, 4, 0] = 1    # 01 -> 00 carry for addition one further (3)
    last_node[1, 2, 4, 3] = 1    # 10 -> 11 carry for addition (1)
    last_node[3, 3, 4, 2] = 1    # 11 -> 10 carry for addition one further (3)
    last_node[4, 0, 4, 1] = 1    # 00 -> 01 carry for subtraction one further (4)
    last_node[2, 1, 4, 0] = 1    # 01 -> 00 carry for subtraction (2)
    last_node[4, 2, 4, 3] = 1    # 10 -> 11 carry for subtraction one further (4)
    last_node[2, 3, 4, 2] = 1    # 11 -> 10 carry for subtraction (2)
    # 5 from right: add and subtract 4
    last_node[3, 0, 5, 0] = 1    # 00 -> 00 carry for addition one further (3)
    last_node[3, 1, 5, 1] = 1    # 01 -> 01 carry for addition one further (3)
    last_node[3, 2, 5, 2] = 1    # 10 -> 10 carry for addition one further (3)
    last_node[3, 3, 5, 3] = 1    # 11 -> 11 carry for addition one further (3)
    last_node[4, 0, 5, 0] = 1    # 00 -> 00 carry for subtraction one further (4)
    last_node[4, 1, 5, 1] = 1    # 01 -> 01 carry for subtraction one further (4)
    last_node[4, 2, 5, 2] = 1    # 10 -> 10 carry for subtraction one further (4)
    last_node[4, 3, 5, 3] = 1    # 11 -> 11 carry for subtraction one further (4)
    # right terminator
    right_terminator = 1/dx**2*cp.array([-205/72, 8/5, 8/5, -1/5, 8/315, -1/560], dtype='float64').reshape(6, 1) # coefficients

    # Define arrays for MPO
    left_node = contract('ab, burd->aurd', left_terminator, central_node, options=options)
    right_node = contract('lurd, re->lued', last_node, right_terminator, options=options)
    arrays = [left_node] + [central_node]*(n-3) + [sec_last_node] + [right_node]

    return arrays


def Diff_2_8_x_MPO(n, dx, options=None):
    # second order derivative with eighth order precision in x direction
    # tensor stucture:
    #                          |                         |               |
    # left_terminator - (central_node)^(n-2) - second to last node - last node - right_terminator
    #                          |                         |               |
    # left terminator
    left_terminator = cp.array([1, 1, 1], dtype='float64').reshape(1, 3)
    # central node
    central_node = cp.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    central_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    central_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    central_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: subtract 1
    central_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for subtraction (2)
    central_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for subtraction (2)
    central_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    central_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # second to last node
    sec_last_node = cp.zeros((3, 4, 5, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    sec_last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    sec_last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    sec_last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    sec_last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    sec_last_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    sec_last_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    sec_last_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    sec_last_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: subtract 1
    sec_last_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for subtraction (2)
    sec_last_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for subtraction (2)
    sec_last_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    sec_last_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # 3 from right: identity with carry for addition
    sec_last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    sec_last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    sec_last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    sec_last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    # 4 from right: identity with carry for subtraction
    sec_last_node[2, 0, 4, 0] =  1   # 00 -> 00 carry for subtraction (2)
    sec_last_node[2, 1, 4, 1] =  1   # 01 -> 01 carry for subtraction (2)
    sec_last_node[2, 2, 4, 2] =  1   # 10 -> 10 carry for subtraction (2)
    sec_last_node[2, 3, 4, 3] =  1   # 11 -> 11 carry for subtraction (2)
    # last node
    last_node = cp.zeros((5, 4, 6, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    last_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    last_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    last_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    last_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: subtract 1
    last_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for subtraction (2)
    last_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for subtraction (2)
    last_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    last_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # 3 from right: add and subtract 2
    last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    last_node[2, 0, 3, 0] = 1    # 00 -> 00 carry for subtraction (2)
    last_node[2, 1, 3, 1] = 1    # 01 -> 01 carry for subtraction (2)
    last_node[2, 2, 3, 2] = 1    # 10 -> 10 carry for subtraction (2)
    last_node[2, 3, 3, 3] = 1    # 11 -> 11 carry for subtraction (2)
    # 4 from right: add and subtract 3
    last_node[1, 0, 4, 2] = 1    # 00 -> 10 carry for addition (1)
    last_node[1, 1, 4, 3] = 1    # 01 -> 11 carry for addition (1)
    last_node[3, 2, 4, 0] = 1    # 10 -> 00 carry for addition one further (3)
    last_node[3, 3, 4, 1] = 1    # 11 -> 01 carry for addition one further (3)
    last_node[4, 0, 4, 2] = 1    # 00 -> 10 carry for subtraction one further (4)
    last_node[4, 1, 4, 3] = 1    # 01 -> 11 carry for subtraction one further (4)
    last_node[2, 2, 4, 0] = 1    # 10 -> 00 carry for subtraction (2)
    last_node[2, 3, 4, 1] = 1    # 11 -> 01 carry for subtraction (2)
    # 5 from right: add and subtract 4
    last_node[3, 0, 5, 0] = 1    # 00 -> 00 carry for addition one further (3)
    last_node[3, 1, 5, 1] = 1    # 01 -> 01 carry for addition one further (3)
    last_node[3, 2, 5, 2] = 1    # 10 -> 10 carry for addition one further (3)
    last_node[3, 3, 5, 3] = 1    # 11 -> 11 carry for addition one further (3)
    last_node[4, 0, 5, 0] = 1    # 00 -> 00 carry for subtraction one further (4)
    last_node[4, 1, 5, 1] = 1    # 01 -> 01 carry for subtraction one further (4)
    last_node[4, 2, 5, 2] = 1    # 10 -> 10 carry for subtraction one further (4)
    last_node[4, 3, 5, 3] = 1    # 11 -> 11 carry for subtraction one further (4)
    # right terminator
    right_terminator = 1/dx**2*cp.array([-205/72, 8/5, 8/5, -1/5, 8/315, -1/560], dtype='float64').reshape(6, 1) # coefficients

    # Define arrays for MPO
    left_node = contract('ab, burd->aurd', left_terminator, central_node, options=options)
    right_node = contract('lurd, re->lued', last_node, right_terminator, options=options)
    arrays = [left_node] + [central_node]*(n-3) + [sec_last_node] + [right_node]

    return arrays