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

import numpy as np
from scipy.sparse.linalg import svds as truncated_svd
from scipy.linalg import svd as plain_svd
from scipy.sparse.linalg import cg
import quimb as qu
import sys
import matplotlib.pyplot as plt
from differential_mpo import *
from differential_operators_numpy import *


# Initial conditions for DJ
def J(X, Y, u_0, y_min=0.4, y_max=0.6, h = 0.005):
    return u_0/2*(np.tanh((Y-y_min)/h)-np.tanh((Y-y_max)/h)-1), np.zeros_like(Y)


def d_1(X, Y, y_min=0.4, y_max=0.6, h=0.005, L_box=1):
    return 2*L_box/h**2*((Y-y_max)*np.exp(-(Y-y_max)**2/h**2)+(Y-y_min)*np.exp(-(Y-y_min)**2/h**2))*(np.sin(8*np.pi*X/L_box)+np.sin(24*np.pi*X/L_box)+np.sin(6*np.pi*X/L_box))


def d_2(X, Y, y_min=0.4, y_max=0.6, h=0.005, L_box=1):
    return np.pi*(np.exp(-(Y-y_max)**2/h**2)+np.exp(-(Y-y_min)**2/h**2))*(8*np.cos(8*np.pi*X/L_box)+24*np.cos(24*np.pi*X/L_box)+6*np.cos(6*np.pi*X/L_box))


def D(X, Y, u_0, y_min, y_max, h, L_box):
    d1 = d_1(X, Y, y_min, y_max, h, L_box)
    d2 = d_2(X, Y, y_min, y_max, h, L_box)
    delta = u_0/(40*np.max(np.sqrt(d1**2+d2**2)))
    return delta*d1, delta*d2


def initial_fields(L, N, y_min, y_max, h, u_max):
    # generate fields according to the initial conditions of the DJ problem
    dx = L/(N-1)    # dx=dy

    # create 2D grid
    x = np.linspace(0, L-dx, N)
    y = np.linspace(0, L-dx, N)
    Y, X = np.meshgrid(y, x)

    # load initial conditions for DJ
    U, V = J(X, Y, u_max, y_min, y_max, h)
    dU, dV = D(X, Y, u_max, y_min, y_max, h, L)
    U = U + dU
    V = V + dV

    return U, V


def get_A_index(binary):
    # get index in original array A
    # binary = sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
    # A_index = sig_1^x sig_2^x ... sig_n_bits^x sig_1^y sig_2^y ... sig_n_bits^y
    return int(binary[::2]+binary[1::2], 2)


def svd(mat, chi=None):
    min_dim = np.min(mat.shape)
    if chi == None or chi >= min_dim:   # plain svd
        U, S, V = plain_svd(mat, full_matrices=False)
        S = np.diag(S)
    else:   # truncated svd
        chi_k = min_dim-1
        if chi < chi_k:
            chi_k = chi
        U, S, V = truncated_svd(mat, chi_k)
        S = np.diag(S)

    return U, S, V


def convert_to_MPS2D(A, chi=None):  
    # converts scalar field to scale-resolved MPS matrices
    Nx, Ny = A.shape            # Get number of points (Nx equals Ny)
    n = int(np.log2(Nx))        # Get number of (qu)bits
    A_vec = A.reshape((1, -1))  # Flatten function
    
    # Reshape into scale resolving representation B
    w = '0'*2*n                                 # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
    B_vec = np.zeros(4**n).reshape((1, -1))     # similar to F but with scale indices

    for _ in range(4**n):
        F_index = get_A_index(w)                # get original index for w
        C_index = int(w, 2)                     # get corresponding index for w
        w = bin(C_index+1)[2:].zfill(2*n)       # set w += 1 in binary
        B_vec[0, C_index] = A_vec[0, F_index]   

    node = B_vec    # set first node of MPS
    MPS = []        # create MPS as list of matrices
    S_mats = []     # create list for singular value matrices 

    for _ in range(n-1):
        m, n = node.shape
        node = node.reshape((4*m, int(n/4)))
        U, S, V = svd(node, chi)        # svd
        MPS.append(U)                   # save U as first node of MPS
        S_mats.append(S)                # save S
        node = np.matmul(S, V)          # create remaining matrix S*V for next expansion step

    m, n = node.shape
    node = node.reshape((4*m, int(n/4)))
    MPS.append(node)    # add last node to MPS

    return MPS


def convert_to_VF2D(MPS):   
    # converts scale-resolved MPS matrices to scalar field
    n_bits = len(MPS)
    N = 2**n_bits
    node_L = MPS[0]
    for i in range(1, n_bits):
        m, n = node_L.shape
        node_R = MPS[i].reshape((n, -1))
        node_L = np.matmul(node_L, node_R)
        m, n = node_L.shape
        node_L = node_L.reshape((4*m, int(n/4)))
    B_vec = node_L.reshape((1, -1)) 

    w = '0'*2*n_bits                            # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
    A_vec = np.zeros(4**n_bits).reshape((1, -1))     # similar to B but with dimensional indices

    for _ in range(4**n_bits):
        F_index = get_A_index(w)             
        C_index = int(w, 2)                   
        w = bin(C_index+1)[2:].zfill(2*n_bits)     
        A_vec[0, F_index]  = B_vec[0, C_index]

    return A_vec.reshape((N, N))


def convert_MPS_to_quimb(tensor_list, dim_ls):  
    # converts scale-resolved MPS matrices to quimb MPS
    arrays = []
    for tensor in tensor_list:
        m, n = tensor.shape             # m = dim_left_bond*dim_ls, n = dim_right_bond
        dim_left_bond = int(m/dim_ls)   # dimension of left bond
        dim_right_bond = n              # dimension of right bond
        if dim_left_bond == 1:          # the first tensor as no left bond
            data = tensor.reshape((dim_ls, dim_right_bond)).transpose()
        elif dim_right_bond == 1:       # the last tensor has no right bond
            data = tensor.reshape((dim_left_bond, dim_ls)).transpose()
        else:
            data = tensor.reshape((dim_left_bond, dim_ls, dim_right_bond))
        arrays.append(data)
    
    return qu.tensor.MatrixProductState(arrays, shape='lpr')


def convert_quimb_to_MPS(quimb_MPS):    
    # converts quimb MPS to scale-resolved MPS matrices
    MPS = []
    for i, array in enumerate(quimb_MPS.arrays):
        if i == 0:                      # the first tensor as no left bond
            r, p = array.shape
            MPS.append(array.transpose().reshape((p, r)))
        elif i == quimb_MPS.L-1:        # the last tensor as no right bond
            l, p = array.shape
            MPS.append(array.reshape((l*p, 1)))
        else:
            l, r, p = array.shape
            MPS.append(np.transpose(array, (0, 2, 1)).reshape((l*p, r)))
    
    return MPS


def convert_ls(A):              
    # converts normal scalar field to scale-resolved scalar field
    Nx, Ny = A.shape            # Get number of points (Nx equals Ny)
    n = int(np.log2(Nx))        # Get number of (qu)bits
    A_vec = A.reshape((1, -1))  # Flatten function
    
    # Reshape into scale resolving representation C
    w = '0'*2*n                                 # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
    B_vec = np.zeros(4**n).reshape((1, -1))     # similar to A but with scale indices

    for _ in range(4**n):
        A_index = get_A_index(w)                # get original index for w
        B_index = int(w, 2)                     # get corresponding index for w
        w = bin(B_index+1)[2:].zfill(2*n)       # set w += 1 in binary
        B_vec[0, B_index] = A_vec[0, A_index]   

    return B_vec.reshape((Nx, Ny))


def convert_back(A):            
    # converts scale-resolved scalar field to normal scalar field
    Nx, Ny = A.shape            # Get number of points (Nx equals Ny)
    n = int(np.log2(Nx))        # Get number of (qu)bits
    A_vec = A.reshape((1, -1))  # Flatten function
    
    # Reshape into scale resolving representation C
    w = '0'*2*n                                 # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
    B_vec = np.zeros(4**n).reshape((1, -1))     # similar to A but with scale indices

    for _ in range(4**n):
        A_index = get_A_index(w)                # get original index for w
        B_index = int(w, 2)                     # get corresponding index for w
        w = bin(B_index+1)[2:].zfill(2*n)       # set w += 1 in binary
        B_vec[0, A_index] = A_vec[0, B_index]   

    return B_vec.reshape((Nx, Ny))


def hadamard_product_MPO(a_MPS, chi):
    # prepares as MPO from an MPS to perform a hadamard product with another MPS
    phys_dim = a_MPS.phys_dim(0)    # get the size of each physical index
    k_delta = np.zeros((phys_dim, phys_dim, phys_dim))  # initialize kronecker delta as np.array
    for i in range(phys_dim):
        k_delta[i, i, i] = 1    # only set variables to one where each index is the same
    temp_MPS = a_MPS.copy(deep=True)
    for i in range(temp_MPS.L):
        delta = qu.tensor.Tensor(k_delta, inds=(f'k{i}', f'a{i}', f'b{i}'), tags=f'I{i}')   # create a kronecker delta tensor for each individual leg of the MPS
        temp_MPS = temp_MPS & delta     # connect kronecker delta tensor to MPS
        temp_MPS = temp_MPS ^ f'I{i}'   # contract kronecker delta tensor to MPS

    data = list(temp_MPS.arrays)    # convert TN to list of np.arrays in order to convert it to quimb MPO
    data[0] = data[0].transpose((1, 2, 0))   # reorder indices for first tensor (quimb speciality)
    result_MPO = qu.tensor.MatrixProductOperator(data, shape='udlr')    # create quimb MPO
    result_MPO.compress(max_bond=chi)
    
    return result_MPO   # return the MPO


def get_precontracted_LR_mps_mps(mps_2, mps_1, center=0):
    # prepare precontracted networks for dmrg sweeps
    # mps_1:    o--o-- center --o--o
    #           |  |            |  |
    # mps_2:    o--o-- center --o--o
    #           left            right networks

    n = mps_1.L                 # number of sites
    left_networks = [None]*n    # create a list containing the contracted left network for each site
    right_networks = [None]*n   # create a list containing the contracted right network for each site

    # handle boundary networks
    dummy_t = qu.tensor.Tensor(np.ones((1, 1)), ('A', 'B'))     # create a dummy network consisting of a 1
    left_networks[0] = dummy_t.copy(deep=True)
    right_networks[-1] = dummy_t.copy(deep=True)

    # from left to right
    for i in range(center):
        if i == 0:
            A = mps_1[0].copy(deep=True)
            B = mps_2[0].copy(deep=True)

            tn = A | B
            qu.tensor.connect(A, B, 1, 1)   # connect physical bonds of A and B
            F = tn ^...                     # F is the contracted network

        elif i == n-1:
            A = mps_1[n-1].copy(deep=True)
            B = mps_2[n-1].copy(deep=True)
            F = F.copy(deep=True)
            
            tn = A | F
            qu.tensor.connect(A, F, 0, 0)   # connect the left bond of A with the upper one of F
            AF = tn ^ ...
            tn = AF | B
            qu.tensor.connect(AF, B, 0, 1)  # connect the physical bonds of A and B
            qu.tensor.connect(AF, B, 1, 0)  # connect the lower bond of F with the left one of B
            F = tn ^...
        
        else:
            A = mps_1[i].copy(deep=True)
            B = mps_2[i].copy(deep=True)
            F = F.copy(deep=True)

            tn = A | F
            qu.tensor.connect(A, F, 0, 0)   # connect the left bond of A with the upper one of F
            AF = tn ^ ...
            tn = AF | B
            qu.tensor.connect(AF, B, 1, 2)  # connect the physical bonds of A and B
            qu.tensor.connect(AF, B, 2, 0)  # connect the lower bond of F with the left one of B
            F = tn ^ ...
        
        left_networks[i+1] = F
    
    # from right to left
    for i in range(n-1, center, -1):
        if i == n-1:
            A = mps_1[n-1].copy(deep=True)
            B = mps_2[n-1].copy(deep=True)

            tn = A | B 
            qu.tensor.connect(A, B, 1, 1)   # connect the physical bonds of A and B
            F = tn ^...                     # F is the contracted network


        elif i == 0:
            A = mps_1[0].copy(deep=True)
            B = mps_2[0].copy(deep=True)
            F = F.copy(deep=True)
            
            tn = A | F
            qu.tensor.connect(A, F, 0, 0)   # connect the right bond of A with the upper one of F
            AF = tn ^ ...
            tn = AF | B
            qu.tensor.connect(AF, B, 0, 1)  # connect the physical bonds of A and B
            qu.tensor.connect(AF, B, 1, 0)  # connect the lower bond of F with the right one of B
            F = tn ^...
        
        else:
            A = mps_1[i].copy(deep=True)
            B = mps_2[i].copy(deep=True)
            F = F.copy(deep=True)

            tn = A | F
            qu.tensor.connect(A, F, 1, 0)   # connect the right bond of A with the upper one of F
            AF = tn ^ ...
            tn = AF | B
            qu.tensor.connect(AF, B, 1, 2)  # connect the physical bonds of A and B
            qu.tensor.connect(AF, B, 2, 1)  # connect the lower bond of F with the right one of B
            F = tn ^...
        
        right_networks[i-1] = F

    return left_networks, right_networks


def get_precontracted_LR_mps_mpo(mps_2, mpo, mps_1, center=0):
    # prepare precontracted networks for dmrg sweeps
    # mps_1:    o--o-- center --o--o
    #           |  |            |  |
    # mpo:      0--0--        --0--0
    #           |  |            |  |
    # mps_2:    o--o-- center --o--o
    #           left            right networks

    n = mps_1.L
    left_networks = [None]*n
    right_networks = [None]*n

    # handle boundary networks
    dummy_t = qu.tensor.Tensor(np.ones((1, 1, 1)), ('A', 'W', 'B')) # create a dummy network consisting of a 1
    left_networks[0] = dummy_t.copy(deep=True)
    right_networks[-1] = dummy_t.copy(deep=True)

    # from left to right
    for i in range(center):
        if i == 0:
            A = mps_1[0].copy(deep=True)
            B = mps_2[0].copy(deep=True)
            W = mpo[0].copy(deep=True)

            tn = A | W 
            qu.tensor.connect(A, W, 1, 1)   # connect the corresponding physical bonds of A and W
            AW = tn ^ ...
            tn = AW | B
            qu.tensor.connect(AW, B, 2, 1)  # connect the corresponding physical bonds of W and B
            F = tn ^...                     # F is the contracted network

        elif i == n-1:
            A = mps_1[n-1].copy(deep=True)
            B = mps_2[n-1].copy(deep=True)
            W = mpo[n-1].copy(deep=True)
            F = F.copy(deep=True)

            tn = A | F
            qu.tensor.connect(A, F, 0, 0)   # connect the left bond of A with the upper one of F
            AF = tn ^ ...
            tn = AF | W
            qu.tensor.connect(AF, W, 0, 1)  # connect the corresponding physical bonds of A and W
            qu.tensor.connect(AF, W, 1, 0)  # connect the middle bond of F with the left one of W
            FAW = tn ^...
            tn = FAW | B
            qu.tensor.connect(FAW, B, 1, 1) # connect the corresponding physical bonds of B and W
            qu.tensor.connect(FAW, B, 0, 0) # connect the lower bond of F with the left one of B
            F = tn ^...
        
        else:
            A = mps_1[i].copy(deep=True)
            B = mps_2[i].copy(deep=True)
            W = mpo[i].copy(deep=True)
            F = F.copy(deep=True)

            tn = A | F
            qu.tensor.connect(A, F, 0, 0)   # connect the left bond of A with the upper one of F
            AF = tn ^ ...
            tn = AF | W
            qu.tensor.connect(AF, W, 1, 2)  # connect the corresponding physical bonds of A and W
            qu.tensor.connect(AF, W, 2, 0)  # connect the middle bond of F with the left one of W
            FAW = tn ^ ...
            tn = FAW | B
            qu.tensor.connect(FAW, B, 3, 2) # connect the corresponding physical bonds of B and W
            qu.tensor.connect(FAW, B, 1, 0) # connect the lower bond of F with the left one of B
            F = tn ^...
        
        left_networks[i+1] = F

    # from right to left
    for i in range(n-1, center, -1):
        if i == n-1:
            A = mps_1[n-1].copy(deep=True)
            B = mps_2[n-1].copy(deep=True)
            W = mpo[n-1].copy(deep=True)

            tn = A | W 
            qu.tensor.connect(A, W, 1, 1)   # connect the corresponding physical bonds of A and W
            AW = tn ^ ...
            tn = AW | B
            qu.tensor.connect(AW, B, 2, 1)  # connect the corresponding physical bonds of W and B
            F = tn ^...                     # F is the contracted network

        elif i == 0:
            A = mps_1[0].copy(deep=True)
            B = mps_2[0].copy(deep=True)
            W = mpo[0].copy(deep=True)
            F = F.copy(deep=True)

            tn = A | F
            qu.tensor.connect(A, F, 0, 0)   # connect the right bond of A with the upper one of F
            AF = tn ^ ...
            tn = AF | W
            qu.tensor.connect(AF, W, 0, 1)  # connect the corresponding physical bonds of A and W
            qu.tensor.connect(AF, W, 1, 0)  # connect the middle bond of F with the right one of W
            FAW = tn ^...
            tn = FAW | B
            qu.tensor.connect(FAW, B, 1, 1) # connect the corresponding physical bonds of B and W
            qu.tensor.connect(FAW, B, 0, 0) # connect the lower bond of F with the right one of B
            F = tn ^...
        
        else:
            A = mps_1[i].copy(deep=True)
            B = mps_2[i].copy(deep=True)
            W = mpo[i].copy(deep=True)
            F = F.copy(deep=True)

            tn = A | F
            qu.tensor.connect(A, F, 1, 0)   # connect the right bond of A with the upper one of F
            AF = tn ^ ...
            tn = AF | W
            qu.tensor.connect(AF, W, 1, 2)  # connect the corresponding physical bonds of A and W
            qu.tensor.connect(AF, W, 2, 1)  # connect the middle bond of F with the right one of W
            FAW = tn ^ ...
            tn = FAW | B
            qu.tensor.connect(FAW, B, 3, 2) # connect the corresponding physical bonds of B and W
            qu.tensor.connect(FAW, B, 1, 1) # connect the lower bond of F with the right one of B
            F = tn ^...
        
        right_networks[i-1] = F

    return left_networks, right_networks


def update_precontracted_LR_mps_mps(F, B, A, LR):
    # update the precontracted networks for dmrg sweeps
    #                        F--A--
    # For LR='L' contract :  F  |
    #                        F--B--
    #
    #                        --A--F
    # For LR='R' contract :    |  F
    #                        --B--F

    A = A.copy(deep=True)
    B = B.copy(deep=True)
    F = F.copy(deep=True)

    if LR == 'L':
        tn = A | F
        qu.tensor.connect(A, F, 0, 0)   # connect the left bond of A with the upper one of F
        AF = tn ^ ...
        tn = AF | B
        qu.tensor.connect(AF, B, 1, 2)  # connect the physical bonds of A and B
        qu.tensor.connect(AF, B, 2, 0)  # connect the lower bond of F with the left one of B
        F = tn ^ ...
    elif LR == 'R':
        tn = A | F
        qu.tensor.connect(A, F, 1, 0)   # connect the right bond of A with the upper one of B
        AF = tn ^ ...
        tn = AF | B
        qu.tensor.connect(AF, B, 1, 2)  # connect the physical bonds of A and B
        qu.tensor.connect(AF, B, 2, 1)  # connect the lower bond of F with the right one of B
        F = tn ^...
    
    return F


def update_precontracted_LR_mps_mpo(F, B, W, A, LR):
    # update the precontracted networks for dmrg sweeps
    #                        F--A--
    #                        F  |
    # For LR='L' contract :  F--W--
    #                        F  |
    #                        F--B--
    #
    #                        --A--F
    #                          |  F
    # For LR='R' contract :  --W--F
    #                          |  F
    #                        --B--F

    A = A.copy(deep=True)
    B = B.copy(deep=True)
    W = W.copy(deep=True)
    F = F.copy(deep=True)

    if LR == 'L':
        tn = A | F
        qu.tensor.connect(A, F, 0, 0)   # connect the left bond of A with the upper one of F
        AF = tn ^ ...
        tn = AF | W
        qu.tensor.connect(AF, W, 1, 2)  # connect the corresponding physical bonds of A and W
        qu.tensor.connect(AF, W, 2, 0)  # connect the middle bond of F with the left one of W
        FAW = tn ^ ...
        tn = FAW | B
        qu.tensor.connect(FAW, B, 3, 2) # connect the corresponding physical bonds of W and B
        qu.tensor.connect(FAW, B, 1, 0) # connect the lower bond of F with the left one of B
        F = tn ^...

    elif LR == 'R':
        tn = A | F
        qu.tensor.connect(A, F, 1, 0)   # connect the right bond of A with the upper one of F
        AF = tn ^ ...
        tn = AF | W
        qu.tensor.connect(AF, W, 1, 2)  # connect the corresponding physical bonds of A and W
        qu.tensor.connect(AF, W, 2, 1)  # connect the middle bond of F with the right one of W
        FAW = tn ^ ...
        tn = FAW | B
        qu.tensor.connect(FAW, B, 3, 2) # connect the corresponding physical bonds of W and B
        qu.tensor.connect(FAW, B, 1, 1) # connect the lower bond of F with the right one of B
        F = tn ^...
    
    return F


def extract_tensor(tn, i):
    # extract the tensor at site i of an mps

    if i == 0:  # add a dummy left bond 
        tensor = tn[0].copy(deep=True)
        tensor.new_ind('dummy', 1, 0)
    elif i == tn.L-1:   # add a dummy right bond
        tensor = tn[-1].copy(deep=True)
        tensor.new_ind('dummy', 1, 1)
    else:
        tensor = tn[i].copy(deep=True)
    
    return tensor


def copy_mps(MPS):
    # create a simple copy of the mps with new index names
    return qu.tensor.MatrixProductState(MPS.arrays)


def cost_function(U, V, Ax_MPS, Ay_MPS, Bx_MPS, By_MPS, chi, dt, Re, mu, n, dx):
    # cost function (only for debugging purposes)

    d1x = Diff_1_8_x_MPO(n, dx)
    d1y = Diff_1_8_y_MPO(n, dx)
    d2x = Diff_2_8_x_MPO(n, dx)
    d2y = Diff_2_8_y_MPO(n, dx)
    
    # create MPOs for convection-diffusion terms
    Bx_MPO = hadamard_product_MPO(copy_mps(Bx_MPS), chi)
    Bxd1x = Bx_MPO.apply(d1x, compress=True)
    d1xBx = d1x.apply(Bx_MPO, compress=True)
    By_MPO = hadamard_product_MPO(copy_mps(By_MPS), chi)
    Byd1y = By_MPO.apply(d1y, compress=True)
    d1yBy = d1y.apply(By_MPO, compress=True)

    cost_c = 0

    # continuity equation
    continuity_state = (d1x.apply(U, compress=True, max_bond=chi) + d1y.apply(V, compress=True, max_bond=chi))
    cost_c += mu * (continuity_state.H @ continuity_state)

    cost_m = 0

    # momentum equation for x
    momentum_x = (U-Ax_MPS)/dt
    momentum_x += 0.5 * Bxd1x.apply(Bx_MPS, compress=True, max_bond=chi)
    momentum_x += 0.5 * d1xBx.apply(Bx_MPS, compress=True, max_bond=chi)
    momentum_x += 0.5 * Byd1y.apply(Bx_MPS, compress=True, max_bond=chi)
    momentum_x += 0.5 * d1yBy.apply(Bx_MPS, compress=True, max_bond=chi)
    momentum_x += -1/Re * d2x.apply(Bx_MPS, compress=True, max_bond=chi)
    momentum_x += -1/Re * d2y.apply(Bx_MPS, compress=True, max_bond=chi)

    cost_m += momentum_x.H @ momentum_x

    # momentum equation for y
    momentum_y = (V-Ay_MPS)/dt
    momentum_y += 0.5 * Bxd1x.apply(By_MPS, compress=True, max_bond=chi)
    momentum_y += 0.5 * d1xBx.apply(By_MPS, compress=True, max_bond=chi)
    momentum_y += 0.5 * Byd1y.apply(By_MPS, compress=True, max_bond=chi)
    momentum_y += 0.5 * d1yBy.apply(By_MPS, compress=True, max_bond=chi)
    momentum_y += -1/Re * d2x.apply(By_MPS, compress=True, max_bond=chi)
    momentum_y += -1/Re * d2y.apply(By_MPS, compress=True, max_bond=chi)

    cost_m += momentum_y.H @ momentum_y

    return cost_c+cost_m, cost_c, cost_m


# helper functions for the conjugate gradient algorithm in MPS form
def Hx(H_left, H_right, x, W, transpose=False):
    # gives tensor corresponding to H*x
    H_left = H_left.copy(deep=True)
    H_right = H_right.copy(deep=True)
    x = x.copy(deep=True)
    W = W.copy(deep=True)

    tn = H_left | W | H_right | x 
    if transpose:
        qu.tensor.connect(W, H_left, 0, 1)   # connect the left bond of W with the middle one of H_left
        qu.tensor.connect(W, H_right, 1, 1)  # connect the right bond of W with the middle one of H_right
        qu.tensor.connect(x, H_left, 0, 2)   # connect the left bond of x with the lower bond of H_left
        qu.tensor.connect(x, H_right, 2, 2)  # connect the right bond of x with the lower bond of H_right
        qu.tensor.connect(x, W, 1, 3)        # connect the middle bond of x with the lower middle bond of W
    else:
        qu.tensor.connect(W, H_left, 0, 1)   # connect the left bond of W with the middle one of H_left
        qu.tensor.connect(W, H_right, 1, 1)  # connect the right bond of W with the middle one of H_right
        qu.tensor.connect(x, H_left, 0, 0)   # connect the left bond of x with the upper bond of H_left
        qu.tensor.connect(x, H_right, 2, 0)  # connect the right bond of x with the upper bond of H_right
        qu.tensor.connect(x, W, 1, 2)        # connect the middle bond of x with the upper middle bond of W

    o = ((0, 3), (0, 2), (0, 1))
    temp =  tn.contract(optimize=o)
    temp.reindex({temp.inds[0]: 'l', temp.inds[1]: 'p', temp.inds[2]: 'r'}, inplace=True)

    return temp


def Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y):
    # gives tensors corresponding to A*x
    # A = (1 - H_11, -H_12)
    #     (-H_21, 1 - H_22)
    Ax_1 = x_1 - Hx(H_11_left, H_11_right, x_1, d1x_d1x) - Hx(H_12_left, H_12_right, x_2, d1x_d1y)
    Ax_2 = x_2 - Hx(H_12_left, H_12_right, x_1, d1x_d1y, transpose=True) - Hx(H_22_left, H_22_right, x_2, d1y_d1y)

    return Ax_1, Ax_2


def b_Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, b_1, b_2, d1x_d1x, d1x_d1y, d1y_d1y):
    # gives tensors corresponding to b - A*x
    Ax_1, Ax_2 = Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y)

    return b_1 - Ax_1, b_2 - Ax_2


def yAx(y_1, y_2, H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y):
    # gives tensors corresponding to y*A*x
    Ax_1, Ax_2 = Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y)

    return y_1 @ Ax_1 + y_2 @ Ax_2


def solve_LS_cg(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, b_1, b_2):
    r_1, r_2 = b_Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, b_1, b_2, d1x_d1x, d1x_d1y, d1y_d1y)
    p_1 = r_1
    p_2 = r_2

    r_r = r_1 @ r_1 + r_2 @ r_2

    iter = 0
    max_iter = 100
    while r_r > 1e-5 and iter < max_iter:
        iter += 1
        alpha = r_r / yAx(p_1, p_2, H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, p_1, p_2, d1x_d1x, d1x_d1y, d1y_d1y)

        x_1 = x_1 + alpha * p_1
        x_2 = x_2 + alpha * p_2

        r_new_1, r_new_2 = b_Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, b_1, b_2, d1x_d1x, d1x_d1y, d1y_d1y)
        r_new_r_new = r_new_1 @ r_new_1 + r_new_2 @ r_new_2
        beta = r_new_r_new / r_r

        p_1 = r_new_1 + beta * p_1
        p_2 = r_new_2 + beta * p_2

        r_r = r_new_r_new

    return x_1.transpose('l', 'r', 'p').data, x_2.transpose('l', 'r', 'p').data


# linear system solver via matrix inversion in MPS form
def solve_LS_inv(H_11, H_12, H_22, b_1, b_2):
    b_1 = b_1.fuse({'lrp': (b_1.inds[0], b_1.inds[2], b_1.inds[1])}).data
    b_2 = b_2.fuse({'lrp': (b_2.inds[0], b_2.inds[2], b_2.inds[1])}).data
    H_11 = H_11.fuse({'lrp': (H_11.inds[1], H_11.inds[5], H_11.inds[3]), 'wep': (H_11.inds[0], H_11.inds[4], H_11.inds[2])}).data
    H_12 = H_12.fuse({'lrp': (H_12.inds[1], H_12.inds[5], H_12.inds[3]), 'wep': (H_12.inds[0], H_12.inds[4], H_12.inds[2])}).data
    H_22 = H_22.fuse({'lrp': (H_22.inds[1], H_22.inds[5], H_22.inds[3]), 'wep': (H_22.inds[0], H_22.inds[4], H_22.inds[2])}).data
    
    H = np.block([[H_11, H_12], [H_12.T, H_22]])
    A = np.eye(len(H)) - H
    b = np.concatenate((b_1, b_2))
    
    x = np.linalg.solve(A,b)
    U_new, V_new = np.array_split(x, 2)

    return U_new, V_new


# linear system solver via scipy.cg in MPS form
def solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2):
    b_1 = b_1.fuse({'lrp': (b_1.inds[0], b_1.inds[2], b_1.inds[1])}).data
    b_2 = b_2.fuse({'lrp': (b_2.inds[0], b_2.inds[2], b_2.inds[1])}).data
    x_1 = x_1.fuse({'lrp': (x_1.inds[0], x_1.inds[2], x_1.inds[1])}).data
    x_2 = x_2.fuse({'lrp': (x_2.inds[0], x_2.inds[2], x_2.inds[1])}).data
    H_11 = H_11.fuse({'lrp': (H_11.inds[1], H_11.inds[5], H_11.inds[3]), 'wep': (H_11.inds[0], H_11.inds[4], H_11.inds[2])}).data
    H_12 = H_12.fuse({'lrp': (H_12.inds[1], H_12.inds[5], H_12.inds[3]), 'wep': (H_12.inds[0], H_12.inds[4], H_12.inds[2])}).data
    H_22 = H_22.fuse({'lrp': (H_22.inds[1], H_22.inds[5], H_22.inds[3]), 'wep': (H_22.inds[0], H_22.inds[4], H_22.inds[2])}).data
    
    H = np.block([[H_11, H_12], [H_12.T, H_22]])
    A = np.eye(len(H)) - H
    b = np.concatenate((b_1, b_2))
    x = np.concatenate((x_1, x_2))

    x_sol = cg(A, b, x)
    U_new, V_new = np.array_split(x_sol[0], 2)

    return U_new, V_new


# time stepping function
def single_time_step(U, V, Ax_MPS, Ay_MPS, Bx_MPS, By_MPS, chi_mpo, dt, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver='cg'):
    
    # initialize precontracted left and right networks
    U_Ax_left, U_Ax_right = get_precontracted_LR_mps_mps(U, Ax_MPS, 0)
    V_Ay_left, V_Ay_right = get_precontracted_LR_mps_mps(V, Ay_MPS, 0)

    n = U.L # size of MPS

    # create MPOs for convection-diffusion terms
    Bx_MPO = hadamard_product_MPO(copy_mps(Bx_MPS), chi_mpo)
    Bxd1x = Bx_MPO.apply(d1x, compress=True, max_bond=chi_mpo)
    d1xBx = d1x.apply(Bx_MPO, compress=True, max_bond=chi_mpo)
    By_MPO = hadamard_product_MPO(copy_mps(By_MPS), chi_mpo)
    Byd1y = By_MPO.apply(d1y, compress=True, max_bond=chi_mpo)
    d1yBy = d1y.apply(By_MPO, compress=True, max_bond=chi_mpo)

    # convection-diffusion terms for x direction (prefactors not included)
    U_d2x_Bx_left, U_d2x_Bx_right = get_precontracted_LR_mps_mpo(U, d2x, Bx_MPS, 0)
    U_d2y_Bx_left, U_d2y_Bx_right = get_precontracted_LR_mps_mpo(U, d2y, Bx_MPS, 0)
    U_Bxd1x_Bx_left, U_Bxd1x_Bx_right = get_precontracted_LR_mps_mpo(U, Bxd1x, Bx_MPS, 0)
    U_d1xBx_Bx_left, U_d1xBx_Bx_right = get_precontracted_LR_mps_mpo(U, d1xBx, Bx_MPS, 0)
    U_Byd1y_Bx_left, U_Byd1y_Bx_right = get_precontracted_LR_mps_mpo(U, Byd1y, Bx_MPS, 0)
    U_d1yBy_Bx_left, U_d1yBy_Bx_right = get_precontracted_LR_mps_mpo(U, d1yBy, Bx_MPS, 0)

    # convection-diffusion terms for y direction (prefactors not included)
    V_d2x_By_left, V_d2x_By_right = get_precontracted_LR_mps_mpo(V, d2x, By_MPS, 0)
    V_d2y_By_left, V_d2y_By_right = get_precontracted_LR_mps_mpo(V, d2y, By_MPS, 0)
    V_Bxd1x_By_left, V_Bxd1x_By_right = get_precontracted_LR_mps_mpo(V, Bxd1x, By_MPS, 0)
    V_d1xBx_By_left, V_d1xBx_By_right = get_precontracted_LR_mps_mpo(V, d1xBx, By_MPS, 0)
    V_Byd1y_By_left, V_Byd1y_By_right = get_precontracted_LR_mps_mpo(V, Byd1y, By_MPS, 0)
    V_d1yBy_By_left, V_d1yBy_By_right = get_precontracted_LR_mps_mpo(V, d1yBy, By_MPS, 0)

    epsilon = 1e-5              # convergence criterion
    E_0 = 1e-10                 # initialize energy before
    # E_1 = 2*epsilon             # initialize energy after
    E_1 = U @ U + V @ V

    # helper function to compute convection-diffusion terms
    def conv_diff(left_tn, right_tn, A_t, W_t):
        left_tn = left_tn.copy(deep=True)
        right_tn = right_tn.copy(deep=True)
        A_t = A_t.copy(deep=True)
        W_t = W_t.copy(deep=True)

        tn = left_tn | A_t | W_t | right_tn
        qu.tensor.connect(left_tn, A_t, 0, 0)   # connect the upper bond of left_tn with the left one of A_t
        qu.tensor.connect(right_tn, A_t, 0, 1)  # connect the upper bond of right_tn with the right one of A_t
        qu.tensor.connect(W_t, A_t, 2, 2)       # connect the corresponding physical bonds of W_t and A_t
        qu.tensor.connect(W_t, left_tn, 0, 1)   # connect the left bond of W_t with the middle one of left_tn
        qu.tensor.connect(W_t, right_tn, 1, 1)  # connect the right bond of W_t with the middle one of right_tn
        temp = tn^...

        return temp.reindex({temp.inds[0]: 'l', temp.inds[1]: 'p', temp.inds[2]: 'r'})
    
    def H_terms(left_tn, right_tn, W_t):
        left_tn = left_tn.copy(deep=True)
        right_tn = right_tn.copy(deep=True)
        W_t = W_t.copy(deep=True)

        tn = left_tn | W_t | right_tn
        qu.tensor.connect(W_t, left_tn, 0, 1)   # connect the left bond of W_t with the middle one of left_tn
        qu.tensor.connect(W_t, right_tn, 1, 1)  # connect the right bond of W_t with the middle one of right_tn
        temp = tn^...

        return mu*dt**2 * temp  # multiply with prefactors

    run = 0
    while np.abs((E_1-E_0)/E_0) > epsilon:      # do until the energy does not change anymore
        U_trial = U.copy(deep=True)
        V_trial = V.copy(deep=True)
        U_trial.right_canonize()
        V_trial.right_canonize()
        run += 1

        # sweep through MPS and optimize locally
        for i in range(n-1):      
            # Build linear system Ax = b 
            # Prepare individual tensor nodes
            Ax_MPS_i = extract_tensor(Ax_MPS, i)
            Ay_MPS_i = extract_tensor(Ay_MPS, i)
            Bx_MPS_i = extract_tensor(Bx_MPS, i)
            By_MPS_i = extract_tensor(By_MPS, i)
            d2x_i = extract_tensor(d2x, i)
            d2y_i = extract_tensor(d2y, i)
            Bxd1x_i = extract_tensor(Bxd1x, i)
            d1xBx_i = extract_tensor(d1xBx, i)
            Byd1y_i = extract_tensor(Byd1y, i)
            d1yBy_i = extract_tensor(d1yBy, i)
            d1x_d1x_i = extract_tensor(d1x_d1x, i)
            d1x_d1y_i = extract_tensor(d1x_d1y, i)
            d1y_d1y_i = extract_tensor(d1y_d1y, i)

            # b_1
            left_tn = U_Ax_left[i].copy(deep=True)
            right_tn = U_Ax_right[i].copy(deep=True)
            A_t = Ax_MPS_i.copy(deep=True) 

            tn = left_tn | A_t | right_tn
            qu.tensor.connect(left_tn, A_t, 0, 0)   # connect the upper bond of left_tn with the left one of A_t
            qu.tensor.connect(right_tn, A_t, 0, 1)  # connect the upper bond of right_tn with the right one of A_t
            b_1 = tn ^...
            b_1.reindex({b_1.inds[0]: 'l', b_1.inds[1]: 'p', b_1.inds[2]: 'r'}, inplace=True)

            # convection-diffusion terms
            b_1 += dt/Re * conv_diff(U_d2x_Bx_left[i], U_d2x_Bx_right[i], Bx_MPS_i, d2x_i)
            b_1 += dt/Re * conv_diff(U_d2y_Bx_left[i], U_d2y_Bx_right[i], Bx_MPS_i, d2y_i)
            b_1 += -dt/2 * conv_diff(U_Bxd1x_Bx_left[i], U_Bxd1x_Bx_right[i], Bx_MPS_i, Bxd1x_i)
            b_1 += -dt/2 * conv_diff(U_d1xBx_Bx_left[i], U_d1xBx_Bx_right[i], Bx_MPS_i, d1xBx_i)
            b_1 += -dt/2 * conv_diff(U_Byd1y_Bx_left[i], U_Byd1y_Bx_right[i], Bx_MPS_i, Byd1y_i)
            b_1 += -dt/2 * conv_diff(U_d1yBy_Bx_left[i], U_d1yBy_Bx_right[i], Bx_MPS_i, d1yBy_i)

            # b_2
            left_tn = V_Ay_left[i].copy(deep=True)
            right_tn = V_Ay_right[i].copy(deep=True)
            A_t = Ay_MPS_i.copy(deep=True)

            tn = left_tn | A_t | right_tn
            qu.tensor.connect(left_tn, A_t, 0, 0)   # connect the upper bond of left_tn with the left one of A_t
            qu.tensor.connect(right_tn, A_t, 0, 1)  # connect the upper bond of right_tn with the right one of A_t
            b_2 = tn ^...
            b_2.reindex({b_2.inds[0]: 'l', b_2.inds[1]: 'p', b_2.inds[2]: 'r'}, inplace=True)

            # convection-diffusion terms
            b_2 += dt/Re * conv_diff(V_d2x_By_left[i], V_d2x_By_right[i], By_MPS_i, d2x_i)
            b_2 += dt/Re * conv_diff(V_d2y_By_left[i], V_d2y_By_right[i], By_MPS_i, d2y_i)
            b_2 += -dt/2 * conv_diff(V_Bxd1x_By_left[i], V_Bxd1x_By_right[i], By_MPS_i, Bxd1x_i)
            b_2 += -dt/2 * conv_diff(V_d1xBx_By_left[i], V_d1xBx_By_right[i], By_MPS_i, d1xBx_i)
            b_2 += -dt/2 * conv_diff(V_Byd1y_By_left[i], V_Byd1y_By_right[i], By_MPS_i, Byd1y_i)
            b_2 += -dt/2 * conv_diff(V_d1yBy_By_left[i], V_d1yBy_By_right[i], By_MPS_i, d1yBy_i)

            # solve linear system
            if solver == 'cg':
                x_1 = extract_tensor(U_trial, i)
                x_2 = extract_tensor(V_trial, i)
                x_1.reindex({x_1.inds[0]: 'l', x_1.inds[1]: 'r', x_1.inds[2]: 'p'}, inplace=True).transpose('l', 'p', 'r', inplace=True)
                x_2.reindex({x_2.inds[0]: 'l', x_2.inds[1]: 'r', x_2.inds[2]: 'p'}, inplace=True).transpose('l', 'p', 'r', inplace=True)
                U_new, V_new = solve_LS_cg(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], x_1, x_2, mu*dt**2 * d1x_d1x_i, mu*dt**2 * d1x_d1y_i, mu*dt**2 * d1y_d1y_i, b_1, b_2)
            elif solver == 'inv':
                H_11 = H_terms(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x_i)
                H_12 = H_terms(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y_i)
                H_22 = H_terms(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y_i)

                U_new, V_new = solve_LS_inv(H_11, H_12, H_22, b_1, b_2) 
            elif solver == 'scipy.cg':
                H_11 = H_terms(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x_i)
                H_12 = H_terms(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y_i)
                H_22 = H_terms(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y_i)
                x_1 = extract_tensor(U_trial, i)
                x_2 = extract_tensor(V_trial, i)
                x_1.reindex({x_1.inds[0]: 'l', x_1.inds[1]: 'r', x_1.inds[2]: 'p'}, inplace=True).transpose('l', 'p', 'r', inplace=True)
                x_2.reindex({x_2.inds[0]: 'l', x_2.inds[1]: 'r', x_2.inds[2]: 'p'}, inplace=True).transpose('l', 'p', 'r', inplace=True)

                U_new, V_new = solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2)
            else:
                raise Exception(f"The solver '{solver}' is not known. Please use 'cg', 'inv', or 'scipy.cg' instead.")

            # update MPSs and precontracted networks
            # update MPS
            shape = U_trial[i].shape
            U_trial_data = list(U_trial.arrays)
            U_trial_data[i] = U_new.reshape(shape)
            U_trial = qu.tensor.MatrixProductState(U_trial_data, shape='lrp')
            V_trial_data = list(V_trial.arrays)
            V_trial_data[i] = V_new.reshape(shape)
            V_trial = qu.tensor.MatrixProductState(V_trial_data, shape='lrp')

            # shift canonical center 
            U_trial.shift_orthogonality_center(i, i+1)
            V_trial.shift_orthogonality_center(i, i+1)

            # Extract tensor nodes
            U_trial_i = extract_tensor(U_trial, i)
            U_trial_i_copy = extract_tensor(copy_mps(U_trial), i)
            V_trial_i = extract_tensor(V_trial, i)
            V_trial_i_copy = extract_tensor(copy_mps(V_trial), i)

            # update precontracted networks
            U_Ax_left[i+1] = update_precontracted_LR_mps_mps(U_Ax_left[i], U_trial_i, Ax_MPS_i, 'L')
            V_Ay_left[i+1] = update_precontracted_LR_mps_mps(V_Ay_left[i], V_trial_i, Ay_MPS_i, 'L')

            U_d2x_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_d2x_Bx_left[i], U_trial_i, d2x_i, Bx_MPS_i, 'L')
            U_d2y_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_d2y_Bx_left[i], U_trial_i, d2y_i, Bx_MPS_i, 'L')
            U_Bxd1x_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_Bxd1x_Bx_left[i], U_trial_i, Bxd1x_i, Bx_MPS_i, 'L')
            U_d1xBx_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_d1xBx_Bx_left[i], U_trial_i, d1xBx_i, Bx_MPS_i, 'L')
            U_Byd1y_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_Byd1y_Bx_left[i], U_trial_i, Byd1y_i, Bx_MPS_i, 'L')
            U_d1yBy_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_d1yBy_Bx_left[i], U_trial_i, d1yBy_i, Bx_MPS_i, 'L')

            V_d2x_By_left[i+1] = update_precontracted_LR_mps_mpo(V_d2x_By_left[i], V_trial_i, d2x_i, By_MPS_i, 'L')
            V_d2y_By_left[i+1] = update_precontracted_LR_mps_mpo(V_d2y_By_left[i], V_trial_i, d2y_i, By_MPS_i, 'L')
            V_Bxd1x_By_left[i+1] = update_precontracted_LR_mps_mpo(V_Bxd1x_By_left[i], V_trial_i, Bxd1x_i, By_MPS_i, 'L')
            V_d1xBx_By_left[i+1] = update_precontracted_LR_mps_mpo(V_d1xBx_By_left[i], V_trial_i, d1xBx_i, By_MPS_i, 'L')
            V_Byd1y_By_left[i+1] = update_precontracted_LR_mps_mpo(V_Byd1y_By_left[i], V_trial_i, Byd1y_i, By_MPS_i, 'L')
            V_d1yBy_By_left[i+1] = update_precontracted_LR_mps_mpo(V_d1yBy_By_left[i], V_trial_i, d1yBy_i, By_MPS_i, 'L')

            U_d1x_d1x_U_left[i+1] = update_precontracted_LR_mps_mpo(U_d1x_d1x_U_left[i], U_trial_i, d1x_d1x_i, U_trial_i_copy, 'L')
            U_d1x_d1y_V_left[i+1] = update_precontracted_LR_mps_mpo(U_d1x_d1y_V_left[i], U_trial_i, d1x_d1y_i, V_trial_i, 'L')
            V_d1y_d1y_V_left[i+1] = update_precontracted_LR_mps_mpo(V_d1y_d1y_V_left[i], V_trial_i, d1y_d1y_i, V_trial_i_copy, 'L')
        
        # sweep back through MPS and optimize locally
        for i in range(n-1, 0, -1):
            # Build linear system Ax = b 
            # Prepare individual tensor nodes
            Ax_MPS_i = extract_tensor(Ax_MPS, i)
            Ay_MPS_i = extract_tensor(Ay_MPS, i)
            Bx_MPS_i = extract_tensor(Bx_MPS, i)
            By_MPS_i = extract_tensor(By_MPS, i)
            d2x_i = extract_tensor(d2x, i)
            d2y_i = extract_tensor(d2y, i)
            Bxd1x_i = extract_tensor(Bxd1x, i)
            d1xBx_i = extract_tensor(d1xBx, i)
            Byd1y_i = extract_tensor(Byd1y, i)
            d1yBy_i = extract_tensor(d1yBy, i)
            d1x_d1x_i = extract_tensor(d1x_d1x, i)
            d1x_d1y_i = extract_tensor(d1x_d1y, i)
            d1y_d1y_i = extract_tensor(d1y_d1y, i)

            # b_1
            left_tn = U_Ax_left[i].copy(deep=True)
            right_tn = U_Ax_right[i].copy(deep=True)
            A_t = Ax_MPS_i.copy(deep=True) 

            tn = left_tn | A_t | right_tn
            qu.tensor.connect(left_tn, A_t, 0, 0)   # connect the upper bond of left_tn with the left one of A_t
            qu.tensor.connect(right_tn, A_t, 0, 1)  # connect the upper bond of right_tn with the right one of A_t
            b_1 = tn ^...
            b_1.reindex({b_1.inds[0]: 'l', b_1.inds[1]: 'p', b_1.inds[2]: 'r'}, inplace=True)

            # convection-diffusion terms
            b_1 += dt/Re * conv_diff(U_d2x_Bx_left[i], U_d2x_Bx_right[i], Bx_MPS_i, d2x_i)
            b_1 += dt/Re * conv_diff(U_d2y_Bx_left[i], U_d2y_Bx_right[i], Bx_MPS_i, d2y_i)
            b_1 += -dt/2 * conv_diff(U_Bxd1x_Bx_left[i], U_Bxd1x_Bx_right[i], Bx_MPS_i, Bxd1x_i)
            b_1 += -dt/2 * conv_diff(U_d1xBx_Bx_left[i], U_d1xBx_Bx_right[i], Bx_MPS_i, d1xBx_i)
            b_1 += -dt/2 * conv_diff(U_Byd1y_Bx_left[i], U_Byd1y_Bx_right[i], Bx_MPS_i, Byd1y_i)
            b_1 += -dt/2 * conv_diff(U_d1yBy_Bx_left[i], U_d1yBy_Bx_right[i], Bx_MPS_i, d1yBy_i)

            # b_2
            left_tn = V_Ay_left[i].copy(deep=True)
            right_tn = V_Ay_right[i].copy(deep=True)
            A_t = Ay_MPS_i.copy(deep=True)

            tn = left_tn | A_t | right_tn
            qu.tensor.connect(left_tn, A_t, 0, 0)   # connect the upper bond of left_tn with the left one of A_t
            qu.tensor.connect(right_tn, A_t, 0, 1)  # connect the upper bond of right_tn with the right one of A_t
            b_2 = tn ^...
            b_2.reindex({b_2.inds[0]: 'l', b_2.inds[1]: 'p', b_2.inds[2]: 'r'}, inplace=True)

            # convection-diffusion terms
            b_2 += dt/Re * conv_diff(V_d2x_By_left[i], V_d2x_By_right[i], By_MPS_i, d2x_i)
            b_2 += dt/Re * conv_diff(V_d2y_By_left[i], V_d2y_By_right[i], By_MPS_i, d2y_i)
            b_2 += -dt/2 * conv_diff(V_Bxd1x_By_left[i], V_Bxd1x_By_right[i], By_MPS_i, Bxd1x_i)
            b_2 += -dt/2 * conv_diff(V_d1xBx_By_left[i], V_d1xBx_By_right[i], By_MPS_i, d1xBx_i)
            b_2 += -dt/2 * conv_diff(V_Byd1y_By_left[i], V_Byd1y_By_right[i], By_MPS_i, Byd1y_i)
            b_2 += -dt/2 * conv_diff(V_d1yBy_By_left[i], V_d1yBy_By_right[i], By_MPS_i, d1yBy_i)

            # solve linear system
            if solver == 'cg':
                x_1 = extract_tensor(U_trial, i)
                x_2 = extract_tensor(V_trial, i)
                x_1.reindex({x_1.inds[0]: 'l', x_1.inds[1]: 'r', x_1.inds[2]: 'p'}, inplace=True).transpose('l', 'p', 'r', inplace=True)
                x_2.reindex({x_2.inds[0]: 'l', x_2.inds[1]: 'r', x_2.inds[2]: 'p'}, inplace=True).transpose('l', 'p', 'r', inplace=True)
                U_new, V_new = solve_LS_cg(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], x_1, x_2, mu*dt**2 * d1x_d1x_i, mu*dt**2 * d1x_d1y_i, mu*dt**2 * d1y_d1y_i, b_1, b_2)
            elif solver == 'inv':
                H_11 = H_terms(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x_i)
                H_12 = H_terms(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y_i)
                H_22 = H_terms(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y_i)

                U_new, V_new = solve_LS_inv(H_11, H_12, H_22, b_1, b_2) 
            elif solver == 'scipy.cg':
                H_11 = H_terms(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x_i)
                H_12 = H_terms(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y_i)
                H_22 = H_terms(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y_i)
                x_1 = extract_tensor(U_trial, i)
                x_2 = extract_tensor(V_trial, i)
                x_1.reindex({x_1.inds[0]: 'l', x_1.inds[1]: 'r', x_1.inds[2]: 'p'}, inplace=True).transpose('l', 'p', 'r', inplace=True)
                x_2.reindex({x_2.inds[0]: 'l', x_2.inds[1]: 'r', x_2.inds[2]: 'p'}, inplace=True).transpose('l', 'p', 'r', inplace=True)

                U_new, V_new = solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2)
            else:
                raise Exception(f"The solver '{solver}' is not known. Please use 'cg', 'inv', or 'scipy.cg' instead.")

            # update MPSs and precontracted networks
            # update MPS
            shape = U_trial[i].shape
            U_trial_data = list(U_trial.arrays)
            U_trial_data[i] = U_new.reshape(shape)
            U_trial = qu.tensor.MatrixProductState(U_trial_data, shape='lrp')
            V_trial_data = list(V_trial.arrays)
            V_trial_data[i] = V_new.reshape(shape)
            V_trial = qu.tensor.MatrixProductState(V_trial_data, shape='lrp')

            # shift canonical center 
            U_trial.shift_orthogonality_center(i, i-1)
            V_trial.shift_orthogonality_center(i, i-1)

            # Extract tensor nodes
            U_trial_i = extract_tensor(U_trial, i)
            U_trial_i_copy = extract_tensor(copy_mps(U_trial), i)
            V_trial_i = extract_tensor(V_trial, i)
            V_trial_i_copy = extract_tensor(copy_mps(V_trial), i)

            # update precontracted networks
            U_Ax_right[i-1] = update_precontracted_LR_mps_mps(U_Ax_right[i], U_trial_i, Ax_MPS_i, 'R')
            V_Ay_right[i-1] = update_precontracted_LR_mps_mps(V_Ay_right[i], V_trial_i, Ay_MPS_i, 'R')

            U_d2x_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_d2x_Bx_right[i], U_trial_i, d2x_i, Bx_MPS_i, 'R')
            U_d2y_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_d2y_Bx_right[i], U_trial_i, d2y_i, Bx_MPS_i, 'R')
            U_Bxd1x_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_Bxd1x_Bx_right[i], U_trial_i, Bxd1x_i, Bx_MPS_i, 'R')
            U_d1xBx_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_d1xBx_Bx_right[i], U_trial_i, d1xBx_i, Bx_MPS_i, 'R')
            U_Byd1y_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_Byd1y_Bx_right[i], U_trial_i, Byd1y_i, Bx_MPS_i, 'R')
            U_d1yBy_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_d1yBy_Bx_right[i], U_trial_i, d1yBy_i, Bx_MPS_i, 'R')

            V_d2x_By_right[i-1] = update_precontracted_LR_mps_mpo(V_d2x_By_right[i], V_trial_i, d2x_i, By_MPS_i, 'R')
            V_d2y_By_right[i-1] = update_precontracted_LR_mps_mpo(V_d2y_By_right[i], V_trial_i, d2y_i, By_MPS_i, 'R')
            V_Bxd1x_By_right[i-1] = update_precontracted_LR_mps_mpo(V_Bxd1x_By_right[i], V_trial_i, Bxd1x_i, By_MPS_i, 'R')
            V_d1xBx_By_right[i-1] = update_precontracted_LR_mps_mpo(V_d1xBx_By_right[i], V_trial_i, d1xBx_i, By_MPS_i, 'R')
            V_Byd1y_By_right[i-1] = update_precontracted_LR_mps_mpo(V_Byd1y_By_right[i], V_trial_i, Byd1y_i, By_MPS_i, 'R')
            V_d1yBy_By_right[i-1] = update_precontracted_LR_mps_mpo(V_d1yBy_By_right[i], V_trial_i, d1yBy_i, By_MPS_i, 'R')

            U_d1x_d1x_U_right[i-1] = update_precontracted_LR_mps_mpo(U_d1x_d1x_U_right[i], U_trial_i, d1x_d1x_i, U_trial_i_copy, 'R')
            U_d1x_d1y_V_right[i-1] = update_precontracted_LR_mps_mpo(U_d1x_d1y_V_right[i], U_trial_i, d1x_d1y_i, V_trial_i, 'R')
            V_d1y_d1y_V_right[i-1] = update_precontracted_LR_mps_mpo(V_d1y_d1y_V_right[i], V_trial_i, d1y_d1y_i, V_trial_i_copy, 'R')

        E_0 = E_1                            # set the previous "Energy" to E_0
        E_1 = U_trial @ U_trial + V_trial @ V_trial      # calculate the new "Energy" from new states
        U = U_trial.copy(deep=True)
        V = V_trial.copy(deep=True)
        print(f"Run: {run}, Diff: {(E_1-E_0)/E_0}, E_0: {E_0}, E_1: {E_1}", end='\r')
    print(f"Run: {run}, Diff: {(E_1-E_0)/E_0}, E_0: {E_0}, E_1: {E_1}", end='\r')
    
    return U, V


def plot(U, V, time=-1, full=False, save_path=None, show=False):
    # plot velocity field given as MPS
    u = convert_to_VF2D(convert_quimb_to_MPS(U))
    v = convert_to_VF2D(convert_quimb_to_MPS(V))
    n = U.L     

    # Genaral parameters
    N = 2**n                        # number of grid points
    dx = 1 / (N-1)                  # finite spacing
    x = np.linspace(0, 1-dx, N)
    y = np.linspace(0, 1-dx, N)
    Y, X = np.meshgrid(y, x)
    n_s = 2**(n-4)                  # Plot N/n_s number of arrows

    plt.figure()
    plt.contourf(X, Y, Dx(v, dx)-Dy(u, dx), 100, cmap="seismic")
    plt.colorbar()
    if full:
        plt.quiver(X, Y, u, v, color="black")
    else:
        plt.quiver(X[::n_s, ::n_s], Y[::n_s, ::n_s], u[::n_s, ::n_s], v[::n_s, ::n_s], color="black")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(f"Time: {round(time, 5)}")
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


# time evolution algorithm
def time_evolution(U, V, chi_mpo, dt, T, Re, mu, save_path, solver='cg'):
    n = U.L
    dx = 1 / (2**n - 1)
    n_steps = int(np.ceil(T/dt))    # time steps
    # finite difference operators with 8th order precision
    d1x = Diff_1_8_x_MPO(n, dx)
    d1y = Diff_1_8_y_MPO(n, dx)
    d2x = Diff_2_8_x_MPO(n, dx)
    d2y = Diff_2_8_y_MPO(n, dx)

    # finite difference operators with 2nd order precision 
    # d1x = Diff_1_2_x_MPO(n, dx)
    # d1y = Diff_1_2_y_MPO(n, dx)
    # d2x = Diff_2_2_x_MPO(n, dx)
    # d2y = Diff_2_2_y_MPO(n, dx)

    d1x_d1x = d1x.apply(d1x, compress=True)
    d1x_d1y = d1x.apply(d1y, compress=True)
    d1y_d1y = d1y.apply(d1y, compress=True)
    
    # bring the orthogonality center to the first tensor
    U.right_canonize()
    V.right_canonize()

    # initialize precontracted left and right networks
    U_d1x_d1x_U_left, U_d1x_d1x_U_right = get_precontracted_LR_mps_mpo(U, d1x_d1x, copy_mps(U), 0)
    U_d1x_d1y_V_left, U_d1x_d1y_V_right = get_precontracted_LR_mps_mpo(U, d1x_d1y, V, 0)
    V_d1y_d1y_V_left, V_d1y_d1y_V_right = get_precontracted_LR_mps_mpo(V, d1y_d1y, copy_mps(V), 0)

    t = 0
    for step in range(n_steps):   # for every time step dt
        print(f"Step = {step} - Time = {t}", end='\n')
        # if step%20 == 0:
        #     plot(U, V, time=t, save_path=f"{save_path}/step_{step}.png", show=False)
        #     np.save(f"{save_path}/u_step_{step}.npy", np.array(U.arrays, dtype=object))
        #     np.save(f"{save_path}/v_step_{step}.npy", np.array(V.arrays, dtype=object))

        U_trial = copy_mps(U)          # trial velocity state
        V_trial = copy_mps(V)          # trial velocity state

        # U_trial = rand_mps_like(U)
        # V_trial = rand_mps_like(V)

        U_prev = copy_mps(U)           # previous velocity state
        V_prev = copy_mps(V)           # previous velocity state

        U_prev_copy = copy_mps(U)           # previous velocity state
        V_prev_copy = copy_mps(V)           # previous velocity state

        # Midpoint RK-2 step
        U_mid, V_mid = single_time_step(U_trial, V_trial, U_prev, V_prev, U_prev_copy, V_prev_copy, chi_mpo, dt/2, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver=solver)
        # Full RK-2 step
        print('')
        U, V = single_time_step(U_trial, V_trial, U_prev, V_prev, U_mid, V_mid, chi_mpo, dt, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver=solver)
        # U, V = single_time_step(U_trial, V_trial, U_prev, V_prev, U_prev_copy, V_prev_copy, chi_mpo, dt, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right)
        print('\n')
        t += dt
    
    # plot(U, V, time=t, save_path=f"{save_path}/final.png", show=False)
    # np.save(f"{save_path}/u_final.npy", np.array(U.arrays, dtype=object))
    # np.save(f"{save_path}/v_final.npy", np.array(V.arrays, dtype=object))
        


def build_initial_fields(n_bits, L, chi, y_min=0.4, y_max=0.6, h=1/200, u_max=1):
    # Gridpoints per spatial dimension
    N = 2**n_bits

    # Generate initial fields
    U, V = initial_fields(L, N, y_min, y_max, h, u_max) 

    # Rescale into non-dimensional units
    U = U/u_max
    V = V/u_max

    # Convert them to MPS form
    MPS_U = convert_to_MPS2D(U, chi)
    MPS_V = convert_to_MPS2D(V, chi)

    # Tranform into quimb MPS form
    MPS_U_quimb = convert_MPS_to_quimb(MPS_U, 4)
    MPS_V_quimb = convert_MPS_to_quimb(MPS_V, 4)

    return MPS_U_quimb, MPS_V_quimb, MPS_U, MPS_V