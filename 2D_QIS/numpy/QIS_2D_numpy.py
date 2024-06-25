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

# This file is restricted to RK2 time stepping
import numpy as np
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds as truncated_svd
from scipy.linalg import svd as plain_svd
from scipy.linalg import qr, rq
from differential_mpo import *
from differential_operators_numpy import *
import sys


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


def multiply_mps_mpo(mps, mpo, chi=None):
    # peforms mps mpo multiplication
    n = len(mps)
    t = np.einsum('ipj,kplm->ijlm', mps[0], mpo[0])
    output_mps = []
    for i in range(1, n):
        t_mps_mpo = np.einsum('ijlm,jqr,lqsn->imrsn', t, mps[i], mpo[i])
        i_s, m_s, r_s, s_s, n_s = t_mps_mpo.shape
        t_mps_mpo = t_mps_mpo.reshape((i_s*m_s, -1))
        mps_node, S, V = svd(t_mps_mpo, chi)
        t = np.matmul(S, V).reshape((-1, r_s, s_s, n_s))
        output_mps.append(mps_node.reshape((i_s, m_s, -1)))
    t = t.reshape(-1,4,1)
    output_mps.append(t)
    
    return output_mps


def multiply_mpo_mpo(mpo_2, mpo_1, chi=None):
    # peforms mpo mpo multiplication
    n = len(mpo_1)
    t = np.einsum('akbp,lprP->akbrP', mpo_1[0], mpo_2[0])
    output_mpo = []
    for i in range(1, n):
        t_mpo_mpo = np.einsum('akbrP,bKcD,rDeF->akPKceF', t, mpo_1[i], mpo_2[i])
        a_s, k_s, P_s, K_s, c_s, e_s, F_s = t_mpo_mpo.shape
        t_mpo_mpo = t_mpo_mpo.reshape((a_s*k_s*P_s, -1))
        mpo_node, S, V = svd(t_mpo_mpo, chi)
        t = np.matmul(S, V).reshape((-1, K_s, c_s, e_s, F_s))
        output_mpo.append(mpo_node.reshape((a_s, k_s, P_s, -1)).transpose((0, 1, 3, 2)))
    t = t.reshape(-1,4,1,4)
    output_mpo.append(t)
    
    return output_mpo


def canonicalize_mps_tensors(a, b, absorb='right'):
    # Perform canonicalization of two MPS tensors.
    if absorb == 'right':
        i_s, p_s, j_s = a.shape
        a = a.reshape((i_s*p_s, j_s))
        a, r = qr(a)
        a = a.reshape((i_s, p_s, -1))
        b = np.einsum('xj,jpk->xpk', r, b) # combine b with r
    elif absorb == 'left':
        j_s, p_s, k_s = b.shape
        b = b.reshape((j_s, p_s*k_s))
        r, b = rq(b)
        b = b.reshape((-1, p_s, k_s))
        a = np.einsum('jx,ipj->ipx', r, a) # combine a with r 
    else:
        raise ValueError(f"absorb must be either left or right")
    return a, b


def right_canonicalize_mps(mps_tensors, start, end):
    # Perform a in-place canonicalization sweep of MPS from left to right.
    mps_tensors = mps_tensors.copy()
    assert end >= start
    for i in range(start, end):
        mps_tensors[i:i+2] = canonicalize_mps_tensors(*mps_tensors[i:i+2], absorb='right')
    return mps_tensors


def left_canonicalize_mps(mps_tensors, start, end):
    # Perform a in-place canonicalization sweep of MPS from right to left.
    mps_tensors = mps_tensors.copy()
    assert start >= end
    for i in range(start, end, -1):
        mps_tensors[i-1:i+1] = canonicalize_mps_tensors(*mps_tensors[i-1:i+1], absorb='left')
    return mps_tensors


def canonical_center(mps, center):
    # right and left canonicalize mps from scratch
    mps_r = right_canonicalize_mps(mps, 0, center)
    mps_rl = left_canonicalize_mps(mps_r, len(mps)-1, center)

    return mps_rl

def shift_canonical_center(mps, center, initial=None):
    # shifts canonical center from initial site
    if initial == None:
        return canonical_center(mps, center)
    elif initial > center:
        mps = mps.copy()
        for i in range(initial, center, -1):
            mps[i-1:i+1] = canonicalize_mps_tensors(*mps[i-1:i+1], absorb='left')
        return mps
    else:
        mps = mps.copy()
        for i in range(initial, center):
            mps[i:i+2] = canonicalize_mps_tensors(*mps[i:i+2], absorb='right')
        return mps


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


def convert_to_MPS2D(A, chi=None):  
    # converts scalar field to scale-resolved MPS matrices
    Nx, Ny = A.shape            # Get number of points (Nx equals Ny)
    n = int(np.log2(Nx))        # Get number of (qu)bits
    A_vec = A.reshape((1, -1))  # Flatten function
    
    # Reshape into scale resolving representation B
    w = '0'*2*n                                 # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
    B_vec = np.zeros(4**n).reshape((1, -1))     # similar to F but with scale indices

    for _ in range(4**n):
        A_index = get_A_index(w)                # get original index for w
        B_index = int(w, 2)                     # get corresponding index for w
        w = bin(B_index+1)[2:].zfill(2*n)       # set w += 1 in binary
        B_vec[0, B_index] = A_vec[0, A_index]   

    node = B_vec    # set first node of MPS
    MPS = []        # create MPS as list of matrices

    for _ in range(n-1):
        m, n = node.shape
        node = node.reshape((4*m, int(n/4)))
        U, S, V = svd(node, chi)        # svd
        MPS.append(U)                   # save U as first node of MPS
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
        A_index = get_A_index(w)             
        B_index = int(w, 2)                   
        w = bin(B_index+1)[2:].zfill(2*n_bits)     
        A_vec[0, A_index]  = B_vec[0, B_index]

    return A_vec.reshape((N, N))


def convert_MPS_to_numpy(tensor_list, dim_p):
    arrays = []
    for tensor in tensor_list:
        m, n = tensor.shape
        dim_left_bond = int(m/dim_p)
        dim_right_bond = n
        data = tensor.reshape((dim_left_bond, dim_p, dim_right_bond))
        arrays.append(data)
    
    return arrays


def convert_numpy_to_MPS(mps):
    arrays = []
    for tensor in mps:
        l, p, r = tensor.shape
        arrays.append(tensor.reshape((l*p, r)))

    return arrays


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


def hadamard_product_MPO(mps):
    # prepares as MPO from an MPS to perform a hadamard product with another MPS
    # mps:      o--o--o--o--o
    #           |  |  |  |  |
    #
    #           |  |  |  |  |
    # k_delta:  k  k  k  k  k
    #          /\ /\ /\ /\ /\
    #
    # -> mpo:   o--o--o--o--o
    #          /\ /\ /\ /\ /\
    
    k_delta = np.zeros((4, 4, 4), dtype='float64')  # initialize kronecker delta as np.array
    for i in range(4):
        k_delta[i, i, i] = 1    # only set variables to one where each index is the same
    mpo = mps.copy()
    for i, tensor in enumerate(mpo):
        mpo[i] = np.einsum('ijk, jlm->ilkm', tensor, k_delta)
    
    return mpo   # return the MPO


def get_precontracted_LR_mps_mps(mps_2, mps_1, center=0):
    # prepare precontracted networks for dmrg sweeps
    # mps_1:    o--o-- center --o--o
    #           |  |            |  |
    # mps_2:    o--o-- center --o--o
    #           left            right networks

    n = len(mps_1)              # number of sites
    left_networks = [None]*n    # create a list containing the contracted left network for each site
    right_networks = [None]*n   # create a list containing the contracted right network for each site

    # handle boundary networks
    dummy_t = np.ones((1, 1))   # create a dummy network consisting of a 1
    left_networks[0] = dummy_t.copy()
    right_networks[-1] = dummy_t.copy()

    o = ['einsum_path', (0, 2), (0, 1)]

    # from left to right
    for i in range(center):
        A = mps_1[i].copy()
        B = mps_2[i].copy()
        F = left_networks[i].copy()
        F_new = np.einsum('apb, cpd, ac->bd', A, B, F, optimize=o)
        
        left_networks[i+1] = F_new
    
    # from right to left
    for i in range(n-1, center, -1):
        A = mps_1[i].copy()
        B = mps_2[i].copy()
        F = right_networks[i].copy()
        
        F_new = np.einsum('apb, cpd, bd->ac', A, B, F, optimize=o)
        
        right_networks[i-1] = F_new

    return left_networks, right_networks


def get_precontracted_LR_mps_mpo(mps_2, mpo, mps_1, center=0):
    # prepare precontracted networks for dmrg sweeps
    # mps_1:    o--o-- center --o--o
    #           |  |            |  |
    # mpo:      0--0--        --0--0
    #           |  |            |  |
    # mps_2:    o--o-- center --o--o
    #           left            right networks

    n = len(mps_1)              # number of sites
    left_networks = [None]*n    # create a list containing the contracted left network for each site
    right_networks = [None]*n   # create a list containing the contracted right network for each site

    # handle boundary networks
    dummy_t = np.ones((1, 1, 1))   # create a dummy network consisting of a 1
    left_networks[0] = dummy_t.copy()
    right_networks[-1] = dummy_t.copy()

    o = ['einsum_path', (0, 3), (0, 2), (0, 1)]

    # from left to right
    for i in range(center):
        A = mps_1[i].copy()
        B = mps_2[i].copy()
        W = mpo[i].copy()
        F = left_networks[i].copy()
        F_new = np.einsum('apb, lprP, cPd, alc->brd', A, W, B, F, optimize=o)
        
        left_networks[i+1] = F_new

    # from right to left
    for i in range(n-1, center, -1):
        A = mps_1[i].copy()
        B = mps_2[i].copy()
        W = mpo[i].copy()
        F = right_networks[i].copy()
        F_new = np.einsum('apb, lprP, cPd, brd->alc', A, W, B, F, optimize=o)
        
        right_networks[i-1] = F_new

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

    A = A.copy()
    B = B.copy()
    F = F.copy()
    o = ['einsum_path', (0, 2), (0, 1)]
    if LR == 'L':
        F_new = np.einsum('apb, cpd, ac->bd', A, B, F, optimize=o)
    elif LR == 'R':
        F_new = np.einsum('apb, cpd, bd->ac', A, B, F, optimize=o)
    
    return F_new


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

    A = A.copy()
    B = B.copy()
    W = W.copy()
    F = F.copy()
    o = ['einsum_path', (0, 3), (0, 2), (0, 1)]
    if LR == 'L':
        F_new = np.einsum('apb, lprP, cPd, alc->brd', A, W, B, F, optimize=o)

    elif LR == 'R':
        F_new = np.einsum('apb, lprP, cPd, brd->alc', A, W, B, F, optimize=o)
    
    return F_new


# helper functions for the conjugate gradient algorithm in MPS form
def Hx(H_left, H_right, x, W, expr='umd, upr, mpeP, reD->dPD'):
    # gives tensor corresponding to H*x
    H_left = H_left.copy()
    H_right = H_right.copy()
    x = x.copy()
    W = W.copy()
    o = ['einsum_path', (0, 1), (0, 2), (0, 1)]

    return np.einsum(expr, H_left, x, W, H_right, optimize=o) 


def Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y):
    # gives tensors corresponding to A*x
    # A = (1 - H_11, -H_12)
    #     (-H_21, 1 - H_22)
    
    Ax_1 = x_1 - Hx(H_11_left, H_11_right, x_1, d1x_d1x) - Hx(H_12_left, H_12_right, x_2, d1x_d1y)
    Ax_2 = x_2 - Hx(H_12_left, H_12_right, x_1, d1x_d1y, 'umd, dPD, mpeP, reD->upr') - Hx(H_22_left, H_22_right, x_2, d1y_d1y)

    return Ax_1, Ax_2


def b_Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, b_1, b_2, d1x_d1x, d1x_d1y, d1y_d1y):
    # gives tensors corresponding to b - A*x
    Ax_1, Ax_2 = Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y)

    return b_1 - Ax_1, b_2 - Ax_2


def yAx(y_1, y_2, H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y):
    # gives tensors corresponding to y*A*x
    Ax_1, Ax_2 = Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y)
    
    yAx_1 = np.einsum('apb, apb->', y_1, Ax_1)
    yAx_2 = np.einsum('apb, apb->', y_2, Ax_2)

    return yAx_1 + yAx_2


# conjugate gradient algorithm in MPS form
def solve_LS_cg(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, b_1, b_2):
    r_1, r_2 = b_Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, b_1, b_2, d1x_d1x, d1x_d1y, d1y_d1y)
    p_1 = r_1
    p_2 = r_2

    r_r = np.einsum('apb, apb->', r_1, r_1) + np.einsum('apb, apb->', r_2, r_2)

    iter = 0
    n = 2
    for s in b_1.shape:
        n *= s
    max_iter = 10*n
    while r_r > 1e-5 and iter < max_iter:
        iter += 1
        alpha = r_r / yAx(p_1, p_2, H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, p_1, p_2, d1x_d1x, d1x_d1y, d1y_d1y)

        x_1 = x_1 + alpha * p_1
        x_2 = x_2 + alpha * p_2

        r_new_1, r_new_2 = b_Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, b_1, b_2, d1x_d1x, d1x_d1y, d1y_d1y)
        r_new_r_new = np.einsum('apb, apb->', r_new_1, r_new_1) + np.einsum('apb, apb->', r_new_2,  r_new_2)
        beta = r_new_r_new / r_r

        p_1 = r_new_1 + beta * p_1
        p_2 = r_new_2 + beta * p_2

        r_r = r_new_r_new

    return x_1, x_2


# linear system solver via matrix inversion in MPS form
def solve_LS_inv(H_11, H_12, H_22, b_1, b_2):
    shape = b_1.shape
    dim = 1
    for d in shape:
        dim *= d
    b_1 = b_1.flatten()
    b_2 = b_2.flatten()
    H_11 = H_11.reshape((dim, dim))
    H_12 = H_12.reshape((dim, dim))
    H_22 = H_22.reshape((dim, dim))
    
    H = np.block([[H_11, H_12.T], [H_12, H_22]])
    A = np.eye(len(H)) - H
    b = np.concatenate((b_1, b_2))
    
    x = np.linalg.solve(A,b)
    U_new, V_new = np.array_split(x, 2)

    return U_new.reshape(shape), V_new.reshape(shape)


# linear system solver via scipy.cg in MPS form
def solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2):
    shape = x_1.shape
    dim = 1
    for d in shape:
        dim *= d
    b_1 = b_1.flatten()
    b_2 = b_2.flatten()
    x_1 = x_1.flatten()
    x_2 = x_2.flatten()
    H_11 = H_11.reshape((dim, dim))
    H_12 = H_12.reshape((dim, dim))
    H_22 = H_22.reshape((dim, dim))
    
    H = np.block([[H_11, H_12.T], [H_12, H_22]])
    A = np.eye(len(H)) - H
    b = np.concatenate((b_1, b_2))
    x = np.concatenate((x_1, x_2))

    x_sol = cg(A, b, x)
    U_new, V_new = np.array_split(x_sol[0], 2)

    return U_new.reshape(shape), V_new.reshape(shape)


# helper function to compute convection-diffusion terms
def left_right_A_W(left_tn, right_tn, A_t, W_t, contract_string='umd, upr, mpeP, reD->dPD'):
    left_tn = left_tn.copy()
    right_tn = right_tn.copy()
    A_t = A_t.copy()
    W_t = W_t.copy()
    o = ['einsum_path', (0, 1), (0, 2), (0, 1)]

    return np.einsum(contract_string, left_tn, A_t, W_t, right_tn, optimize=o)


def left_right_W(left_tn, right_tn, W_t, contract_string='umd, mpeP, reD->uprdPD'):
    left_tn = left_tn.copy()
    right_tn = right_tn.copy()
    W_t = W_t.copy()
    o = ['einsum_path', (0, 1), (0, 1)]

    return np.einsum(contract_string, left_tn, W_t, right_tn, optimize=o)


# time stepping function
def single_time_step(U, V, Ax_MPS, Ay_MPS, Bx_MPS, By_MPS, chi_mpo, dt, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver='cg'):
    
    # initialize precontracted left and right networks
    U_Ax_left, U_Ax_right = get_precontracted_LR_mps_mps(U, Ax_MPS, 0)
    V_Ay_left, V_Ay_right = get_precontracted_LR_mps_mps(V, Ay_MPS, 0)
    
    n = len(U) # size of MPS

    # create MPOs for convection-diffusion terms
    Bx_MPO = hadamard_product_MPO(Bx_MPS)
    Bxd1x = multiply_mpo_mpo(Bx_MPO, d1x, chi_mpo)
    d1xBx = multiply_mpo_mpo(d1x, Bx_MPO, chi_mpo)
    By_MPO = hadamard_product_MPO(By_MPS)
    Byd1y = multiply_mpo_mpo(By_MPO, d1y, chi_mpo)
    d1yBy = multiply_mpo_mpo(d1y, By_MPO, chi_mpo)

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
    E_1 = 2*epsilon             # initialize energy after
    
    run = 0
    while np.abs((E_1-E_0)/E_0) > epsilon:      # do until the state does not change anymore
        run += 1

        # sweep through MPS and optimize locally
        for i in range(n-1):      
            # Build linear system Ax = b 
            # b_1
            left_tn = U_Ax_left[i].copy()
            right_tn = U_Ax_right[i].copy()
            A_t = Ax_MPS[i].copy() 
            b_1 = np.einsum('ud, upr, rD->dpD', left_tn, A_t, right_tn)

            # convection-diffusion terms
            b_1 += dt/Re * left_right_A_W(U_d2x_Bx_left[i], U_d2x_Bx_right[i], Bx_MPS[i], d2x[i])
            b_1 += dt/Re * left_right_A_W(U_d2y_Bx_left[i], U_d2y_Bx_right[i], Bx_MPS[i], d2y[i])
            b_1 += -dt/2 * left_right_A_W(U_Bxd1x_Bx_left[i], U_Bxd1x_Bx_right[i], Bx_MPS[i], Bxd1x[i])
            b_1 += -dt/2 * left_right_A_W(U_d1xBx_Bx_left[i], U_d1xBx_Bx_right[i], Bx_MPS[i], d1xBx[i])
            b_1 += -dt/2 * left_right_A_W(U_Byd1y_Bx_left[i], U_Byd1y_Bx_right[i], Bx_MPS[i], Byd1y[i])
            b_1 += -dt/2 * left_right_A_W(U_d1yBy_Bx_left[i], U_d1yBy_Bx_right[i], Bx_MPS[i], d1yBy[i])
            
            # b_2
            left_tn = V_Ay_left[i].copy()
            right_tn = V_Ay_right[i].copy()
            A_t = Ay_MPS[i].copy()
            b_2 = np.einsum('ud, upr, rD->dpD', left_tn, A_t, right_tn)

            # convection-diffusion terms
            b_2 += dt/Re * left_right_A_W(V_d2x_By_left[i], V_d2x_By_right[i], By_MPS[i], d2x[i])
            b_2 += dt/Re * left_right_A_W(V_d2y_By_left[i], V_d2y_By_right[i], By_MPS[i], d2y[i])
            b_2 += -dt/2 * left_right_A_W(V_Bxd1x_By_left[i], V_Bxd1x_By_right[i], By_MPS[i], Bxd1x[i])
            b_2 += -dt/2 * left_right_A_W(V_d1xBx_By_left[i], V_d1xBx_By_right[i], By_MPS[i], d1xBx[i])
            b_2 += -dt/2 * left_right_A_W(V_Byd1y_By_left[i], V_Byd1y_By_right[i], By_MPS[i], Byd1y[i])
            b_2 += -dt/2 * left_right_A_W(V_d1yBy_By_left[i], V_d1yBy_By_right[i], By_MPS[i], d1yBy[i])

            # solve linear system
            if solver == 'cg':
                U_new, V_new = solve_LS_cg(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], U[i], V[i], mu*dt**2 * d1x_d1x[i], mu*dt**2 * d1x_d1y[i], mu*dt**2 * d1y_d1y[i], b_1, b_2)
            elif solver == 'inv':
                H_11 = mu*dt**2 * left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i])
                H_12 = mu*dt**2 * left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i])
                H_22 = mu*dt**2 * left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i])

                U_new, V_new = solve_LS_inv(H_11, H_12, H_22, b_1, b_2)
            elif solver == 'scipy.cg':
                H_11 = mu*dt**2 * left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i])
                H_12 = mu*dt**2 * left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i])
                H_22 = mu*dt**2 * left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i])
                x_1 = U[i].copy()
                x_2 = V[i].copy()

                U_new, V_new = solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2)
            else:
                raise Exception(f"The solver '{solver}' is not known. Please use 'cg', 'inv', or 'scipy.cg' instead.")

            # update MPSs and precontracted networks
            # update MPS
            U[i] = U_new
            V[i] = V_new

            # shift canonical center 
            U = shift_canonical_center(U, i+1, i)
            V = shift_canonical_center(V, i+1, i)

            # update precontracted networks
            U_Ax_left[i+1] = update_precontracted_LR_mps_mps(U_Ax_left[i], U[i], Ax_MPS[i], 'L')
            V_Ay_left[i+1] = update_precontracted_LR_mps_mps(V_Ay_left[i], V[i], Ay_MPS[i], 'L')

            U_d2x_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_d2x_Bx_left[i], U[i], d2x[i], Bx_MPS[i], 'L')
            U_d2y_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_d2y_Bx_left[i], U[i], d2y[i], Bx_MPS[i], 'L')
            U_Bxd1x_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_Bxd1x_Bx_left[i], U[i], Bxd1x[i], Bx_MPS[i], 'L')
            U_d1xBx_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_d1xBx_Bx_left[i], U[i], d1xBx[i], Bx_MPS[i], 'L')
            U_Byd1y_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_Byd1y_Bx_left[i], U[i], Byd1y[i], Bx_MPS[i], 'L')
            U_d1yBy_Bx_left[i+1] = update_precontracted_LR_mps_mpo(U_d1yBy_Bx_left[i], U[i], d1yBy[i], Bx_MPS[i], 'L')

            V_d2x_By_left[i+1] = update_precontracted_LR_mps_mpo(V_d2x_By_left[i], V[i], d2x[i], By_MPS[i], 'L')
            V_d2y_By_left[i+1] = update_precontracted_LR_mps_mpo(V_d2y_By_left[i], V[i], d2y[i], By_MPS[i], 'L')
            V_Bxd1x_By_left[i+1] = update_precontracted_LR_mps_mpo(V_Bxd1x_By_left[i], V[i], Bxd1x[i], By_MPS[i], 'L')
            V_d1xBx_By_left[i+1] = update_precontracted_LR_mps_mpo(V_d1xBx_By_left[i], V[i], d1xBx[i], By_MPS[i], 'L')
            V_Byd1y_By_left[i+1] = update_precontracted_LR_mps_mpo(V_Byd1y_By_left[i], V[i], Byd1y[i], By_MPS[i], 'L')
            V_d1yBy_By_left[i+1] = update_precontracted_LR_mps_mpo(V_d1yBy_By_left[i], V[i], d1yBy[i], By_MPS[i], 'L')

            U_d1x_d1x_U_left[i+1] = update_precontracted_LR_mps_mpo(U_d1x_d1x_U_left[i], U[i], d1x_d1x[i], U[i], 'L')
            U_d1x_d1y_V_left[i+1] = update_precontracted_LR_mps_mpo(U_d1x_d1y_V_left[i], U[i], d1x_d1y[i], V[i], 'L')
            V_d1y_d1y_V_left[i+1] = update_precontracted_LR_mps_mpo(V_d1y_d1y_V_left[i], V[i], d1y_d1y[i], V[i], 'L')
        
        # sweep back through MPS and optimize locally
        for i in range(n-1, 0, -1):
            # Build linear system Ax = b 
            # b_1
            left_tn = U_Ax_left[i].copy()
            right_tn = U_Ax_right[i].copy()
            A_t = Ax_MPS[i].copy() 
            b_1 = np.einsum('ud, upr, rD->dpD', left_tn, A_t, right_tn)

            # convection-diffusion terms
            b_1 += dt/Re * left_right_A_W(U_d2x_Bx_left[i], U_d2x_Bx_right[i], Bx_MPS[i], d2x[i])
            b_1 += dt/Re * left_right_A_W(U_d2y_Bx_left[i], U_d2y_Bx_right[i], Bx_MPS[i], d2y[i])
            b_1 += -dt/2 * left_right_A_W(U_Bxd1x_Bx_left[i], U_Bxd1x_Bx_right[i], Bx_MPS[i], Bxd1x[i])
            b_1 += -dt/2 * left_right_A_W(U_d1xBx_Bx_left[i], U_d1xBx_Bx_right[i], Bx_MPS[i], d1xBx[i])
            b_1 += -dt/2 * left_right_A_W(U_Byd1y_Bx_left[i], U_Byd1y_Bx_right[i], Bx_MPS[i], Byd1y[i])
            b_1 += -dt/2 * left_right_A_W(U_d1yBy_Bx_left[i], U_d1yBy_Bx_right[i], Bx_MPS[i], d1yBy[i])

            # b_2
            left_tn = V_Ay_left[i].copy()
            right_tn = V_Ay_right[i].copy()
            A_t = Ay_MPS[i].copy()
            b_2 = np.einsum('ud, upr, rD->dpD', left_tn, A_t, right_tn)

            # convection-diffusion terms
            b_2 += dt/Re * left_right_A_W(V_d2x_By_left[i], V_d2x_By_right[i], By_MPS[i], d2x[i])
            b_2 += dt/Re * left_right_A_W(V_d2y_By_left[i], V_d2y_By_right[i], By_MPS[i], d2y[i])
            b_2 += -dt/2 * left_right_A_W(V_Bxd1x_By_left[i], V_Bxd1x_By_right[i], By_MPS[i], Bxd1x[i])
            b_2 += -dt/2 * left_right_A_W(V_d1xBx_By_left[i], V_d1xBx_By_right[i], By_MPS[i], d1xBx[i])
            b_2 += -dt/2 * left_right_A_W(V_Byd1y_By_left[i], V_Byd1y_By_right[i], By_MPS[i], Byd1y[i])
            b_2 += -dt/2 * left_right_A_W(V_d1yBy_By_left[i], V_d1yBy_By_right[i], By_MPS[i], d1yBy[i])

            # solve linear system
            if solver == 'cg':
                U_new, V_new = solve_LS_cg(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], U[i], V[i], mu*dt**2 * d1x_d1x[i], mu*dt**2 * d1x_d1y[i], mu*dt**2 * d1y_d1y[i], b_1, b_2)
            elif solver == 'inv':
                H_11 = mu*dt**2 * left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i])
                H_12 = mu*dt**2 * left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i])
                H_22 = mu*dt**2 * left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i])

                U_new, V_new = solve_LS_inv(H_11, H_12, H_22, b_1, b_2)
            elif solver == 'scipy.cg':
                H_11 = mu*dt**2 * left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i])
                H_12 = mu*dt**2 * left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i])
                H_22 = mu*dt**2 * left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i])
                x_1 = U[i].copy()
                x_2 = V[i].copy()

                U_new, V_new = solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2)
            else:
                raise Exception(f"The solver '{solver}' is not known. Please use 'cg', 'inv', or 'scipy.cg' instead.")

            # update MPSs and precontracted networks
            # update MPS
            U[i] = U_new
            V[i] = V_new

            # shift canonical center 
            U = shift_canonical_center(U, i-1, i)
            V = shift_canonical_center(V, i-1, i)

            # update precontracted networks
            U_Ax_right[i-1] = update_precontracted_LR_mps_mps(U_Ax_right[i], U[i], Ax_MPS[i], 'R')
            V_Ay_right[i-1] = update_precontracted_LR_mps_mps(V_Ay_right[i], V[i], Ay_MPS[i], 'R')

            U_d2x_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_d2x_Bx_right[i], U[i], d2x[i], Bx_MPS[i], 'R')
            U_d2y_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_d2y_Bx_right[i], U[i], d2y[i], Bx_MPS[i], 'R')
            U_Bxd1x_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_Bxd1x_Bx_right[i], U[i], Bxd1x[i], Bx_MPS[i], 'R')
            U_d1xBx_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_d1xBx_Bx_right[i], U[i], d1xBx[i], Bx_MPS[i], 'R')
            U_Byd1y_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_Byd1y_Bx_right[i], U[i], Byd1y[i], Bx_MPS[i], 'R')
            U_d1yBy_Bx_right[i-1] = update_precontracted_LR_mps_mpo(U_d1yBy_Bx_right[i], U[i], d1yBy[i], Bx_MPS[i], 'R')

            V_d2x_By_right[i-1] = update_precontracted_LR_mps_mpo(V_d2x_By_right[i], V[i], d2x[i], By_MPS[i], 'R')
            V_d2y_By_right[i-1] = update_precontracted_LR_mps_mpo(V_d2y_By_right[i], V[i], d2y[i], By_MPS[i], 'R')
            V_Bxd1x_By_right[i-1] = update_precontracted_LR_mps_mpo(V_Bxd1x_By_right[i], V[i], Bxd1x[i], By_MPS[i], 'R')
            V_d1xBx_By_right[i-1] = update_precontracted_LR_mps_mpo(V_d1xBx_By_right[i], V[i], d1xBx[i], By_MPS[i], 'R')
            V_Byd1y_By_right[i-1] = update_precontracted_LR_mps_mpo(V_Byd1y_By_right[i], V[i], Byd1y[i], By_MPS[i], 'R')
            V_d1yBy_By_right[i-1] = update_precontracted_LR_mps_mpo(V_d1yBy_By_right[i], V[i], d1yBy[i], By_MPS[i], 'R')

            U_d1x_d1x_U_right[i-1] = update_precontracted_LR_mps_mpo(U_d1x_d1x_U_right[i], U[i], d1x_d1x[i], U[i], 'R')
            U_d1x_d1y_V_right[i-1] = update_precontracted_LR_mps_mpo(U_d1x_d1y_V_right[i], U[i], d1x_d1y[i], V[i], 'R')
            V_d1y_d1y_V_right[i-1] = update_precontracted_LR_mps_mpo(V_d1y_d1y_V_right[i], V[i], d1y_d1y[i], V[i], 'R')

        E_0 = E_1       
        E_1 = np.einsum('apb, apb->', U[0], U[0]) + np.einsum('apb, apb->', V[0], V[0]) 
        print(f"Run: {run}, Diff: {(E_1-E_0)/E_0}, E_0: {E_0}, E_1: {E_1}", end='\r')
    print(f"Run: {run}, Diff: {(E_1-E_0)/E_0}, E_0: {E_0}, E_1: {E_1}", end='\r')
    
    return U, V


def plot(U, V, time=-1, full=False, save_path=None, show=False):
    # plot velocity field given as MPS
    u = convert_to_VF2D(convert_numpy_to_MPS(U))
    v = convert_to_VF2D(convert_numpy_to_MPS(V))
    n = len(U)    

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
    n = len(U)
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

    d1x_d1x = multiply_mpo_mpo(d1x, d1x, chi_mpo)
    d1x_d1y = multiply_mpo_mpo(d1x, d1y, chi_mpo)
    d1y_d1y = multiply_mpo_mpo(d1y, d1y, chi_mpo)
    
    # bring the orthogonality center to the first tensor
    U = canonical_center(U, 0)
    V = canonical_center(V, 0)

    # initialize precontracted left and right networks
    U_d1x_d1x_U_left, U_d1x_d1x_U_right = get_precontracted_LR_mps_mpo(U, d1x_d1x, U, 0)
    U_d1x_d1y_V_left, U_d1x_d1y_V_right = get_precontracted_LR_mps_mpo(U, d1x_d1y, V, 0)
    V_d1y_d1y_V_left, V_d1y_d1y_V_right = get_precontracted_LR_mps_mpo(V, d1y_d1y, V, 0)

    t = 0
    for step in range(n_steps):   # for every time step dt
        print(f"Step = {step} - Time = {t}", end='\n')
        # if step%20 == 0:
        #     plot(U, V, time=t, save_path=f"{save_path}/step_{step}.png", show=False)
            #np.save(f"{save_path}/u_step_{step}.npy", np.array(U, dtype=object))
            #np.save(f"{save_path}/v_step_{step}.npy", np.array(V, dtype=object))

        U_trial = U.copy()         # trial velocity state
        V_trial = V.copy()         # trial velocity state

        U_prev = U.copy()          # previous velocity state
        V_prev = V.copy()          # previous velocity state

        U_prev_copy = U.copy()          # previous velocity state
        V_prev_copy = V.copy()          # previous velocity state
        
        # Midpoint RK-2 step
        # U_mid, V_mid = single_time_step(U_trial, V_trial, U_prev, V_prev, U_prev_copy, V_prev_copy, chi_mpo, dt/2, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right)
        # Full RK-2 step
        print('')
        # U, V = single_time_step(U_trial, V_trial, U_prev, V_prev, U_mid, V_mid, chi_mpo, dt, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right)
        U, V = single_time_step(U_trial, V_trial, U_prev, V_prev, U_prev_copy, V_prev_copy, chi_mpo, dt, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver=solver)
        print('\n')
        t += dt
    
    # plot(U, V, time=t, save_path=f"{save_path}/final.png", show=False)
    #np.save(f"{save_path}/u_final.npy", np.array(U, dtype=object))
    #np.save(f"{save_path}/v_final.npy", np.array(V, dtype=object))
        


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
    MPS_U_quimb = convert_MPS_to_numpy(MPS_U, 4)
    MPS_V_quimb = convert_MPS_to_numpy(MPS_V, 4)

    return MPS_U_quimb, MPS_V_quimb, MPS_U, MPS_V