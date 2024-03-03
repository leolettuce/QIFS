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

# All functions relevant for the simulation of the 2D TDJ problem with MPS
import numpy as np
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from differential_mpo import *
from differential_operators_numpy import *

import cupy as cp
from cuquantum import cutensornet as cutn
from cuquantum.cutensornet.tensor import decompose, SVDMethod
from cuquantum.cutensornet.experimental import contract_decompose
from cuquantum import contract, tensor, OptimizerOptions, Network
import sys
import json
import time

import subprocess as sp

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


class QI_CFD:

    def __init__(self):
        self.handle = cutn.create()
        self.options = {'handle': self.handle}
        self.networks = {}
        
        self.n_bits = 10
        self.N = 2**self.n_bits
        self.L = 1
        self.chi = 33
        self.chi_mpo = 33
        self.dt = 0.1*2.0**-(self.n_bits-1)
        self.T = 2
        self.dx = 1 / (2**self.n_bits - 1)
        self.mu = 250000.0
        self.Re = 200*1e3
        self.solver = "cg"
        self.path = None
        self.save_number = 100
        self.meas_comp_time = False
        self.comp_time_path = None
        self.meas_cg_iter = False
        self.cg_iter_path = None
        self.cg_iter_data = None

        self.U_init = None
        self.V_init = None

    
    def init_params(self, n_bits, L, chi, chi_mpo, T, mu, Re, path, solver, save_number=100):
        self.n_bits = n_bits
        self.N = 2**self.n_bits
        self.L = L
        self.chi = chi
        self.chi_mpo = chi_mpo
        self.dt = 0.1*2.0**-(self.n_bits-1)
        self.T = T
        self.dx = 1 / (2**self.n_bits - 1)
        self.mu = mu
        self.Re = Re
        self.path = path
        self.solver = solver
        self.save_number = save_number
        print("Initialized parameters")


    def enable_meas_comp_time(self, path):
        self.meas_comp_time = True
        self.comp_time_path = path


    def enable_meas_cg_iter(self, path):
        self.meas_cg_iter = True
        self.cg_iter_path = path


    def multiply_mps_mpo(self, mps, mpo, algorithm, options=None):
        # peforms mps mpo multiplication
        t = contract('ipj,kplm->ijlm', mps[0], mpo[0], options=options)
        output_mps = []
        for i in range(1, self.n_bits):
            mps_node, _, t = contract_decompose('ijlm,jqr,lqsn->imx,xrsn', t, mps[i], mpo[i], algorithm=algorithm, options=options)
            output_mps.append(mps_node)
        t = t.reshape(-1,4,1)
        output_mps.append(t)
        
        return output_mps


    def multiply_mpo_mpo(self, mpo_2, mpo_1, algorithm, options=None):
        # peforms mpo mpo multiplication
        t = contract('akbp,lprP->akbrP', mpo_1[0], mpo_2[0], options=options)
        output_mpo = []
        for i in range(1, self.n_bits):
            mpo, _, t = contract_decompose('akbrP,bKcD,rDeF->akxP,xKceF', t, mpo_1[i], mpo_2[i], algorithm=algorithm, options=options)
            output_mpo.append(mpo)
        t = t.reshape(-1,4,1,4)
        output_mpo.append(t)
        
        return output_mpo


    def canonicalize_mps_tensors(self, a, b, absorb='right', options=None):
        # Perform canonicalization of two MPS tensors.
        if absorb == 'right':
            a, r = tensor.decompose('ipj->ipx,xj', a, options=options) # QR on a
            b = contract('xj,jpk->xpk', r, b, options=options) # combine b with r
        elif absorb == 'left':
            b, r = tensor.decompose('jpk->xpk,jx', b, options=options) # QR on b
            a = contract('jx,ipj->ipx', r, a, options=options) # combine a with r 
        else:
            raise ValueError(f"absorb must be either left or right")
        return a, b


    def right_canonicalize_mps(self, mps_tensors, start, end, options=None):
        # Perform a in-place canonicalization sweep of MPS from left to right.
        assert end >= start
        for i in range(start, end):
            mps_tensors[i:i+2] = self.canonicalize_mps_tensors(*mps_tensors[i:i+2], absorb='right', options=options)
        return mps_tensors


    def left_canonicalize_mps(self, mps_tensors, start, end, options=None):
        # Perform a in-place canonicalization sweep of MPS from right to left.
        assert start >= end
        for i in range(start, end, -1):
            mps_tensors[i-1:i+1] = self.canonicalize_mps_tensors(*mps_tensors[i-1:i+1], absorb='left', options=options)
        return mps_tensors


    def canonical_center(self, mps, center, options=None):
        # right and left canonicalize mps from scratch
        mps_r = self.right_canonicalize_mps(mps, 0, center, options=options)
        mps_rl = self.left_canonicalize_mps(mps_r, self.n_bits-1, center, options=options)

        return mps_rl

    def shift_canonical_center(self, mps, center, initial=None, options=None):
        # shifts canonical center from initial site
        if initial == None:
            return self.canonical_center(mps, center, options)
        elif initial > center:
            for i in range(initial, center, -1):
                mps[i-1:i+1] = self.canonicalize_mps_tensors(*mps[i-1:i+1], absorb='left', options=options)
            return mps
        else:
            for i in range(initial, center):
                mps[i:i+2] = self.canonicalize_mps_tensors(*mps[i:i+2], absorb='right', options=options)
            return mps


    # Initial conditions for TDJ
    def J(self, X, Y, u_0, y_min=0.4, y_max=0.6, h = 0.005):
        return u_0/2*(np.tanh((Y-y_min)/h)-np.tanh((Y-y_max)/h)-1), np.zeros_like(Y)


    def d_1(self, X, Y, y_min=0.4, y_max=0.6, h=0.005, L_box=1):
        return 2*L_box/h**2*((Y-y_max)*np.exp(-(Y-y_max)**2/h**2)+(Y-y_min)*np.exp(-(Y-y_min)**2/h**2))*(np.sin(8*np.pi*X/L_box)+np.sin(24*np.pi*X/L_box)+np.sin(6*np.pi*X/L_box))


    def d_2(self, X, Y, y_min=0.4, y_max=0.6, h=0.005, L_box=1):
        return np.pi*(np.exp(-(Y-y_max)**2/h**2)+np.exp(-(Y-y_min)**2/h**2))*(8*np.cos(8*np.pi*X/L_box)+24*np.cos(24*np.pi*X/L_box)+6*np.cos(6*np.pi*X/L_box))


    def D(self, X, Y, u_0, y_min, y_max, h, L_box):
        d1 = self.d_1(X, Y, y_min, y_max, h, L_box)
        d2 = self.d_2(X, Y, y_min, y_max, h, L_box)
        delta = u_0/(40*np.max(np.sqrt(d1**2+d2**2)))
        return delta*d1, delta*d2


    def initial_fields(self, y_min, y_max, h, u_max):
        # generate fields according to the initial conditions of the 2TDJ problem

        # create 2D grid
        x = np.linspace(0, self.L-self.dx, self.N)
        y = np.linspace(0, self.L-self.dx, self.N)
        Y, X = np.meshgrid(y, x)

        # load initial conditions for TDJ
        U, V = self.J(X, Y, u_max, y_min, y_max, h)
        dU, dV = self.D(X, Y, u_max, y_min, y_max, h, self.L)
        U = U + dU
        V = V + dV

        return U, V


    def get_A_index(self, binary):
        # get index in original array A
        # binary = sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
        # A_index = sig_1^x sig_2^x ... sig_n_bits^x sig_1^y sig_2^y ... sig_n_bits^y
        return int(binary[::2]+binary[1::2], 2)


    def svd(self, mat, chi):
        # Perform truncated singular value decomposition 
        U, S, V = decompose('ij->ik,kj', mat, method=SVDMethod(max_extent=chi))

        return U, np.diag(S), V


    def convert_to_MPS2D(self, A, chi=None):  
        # converts scalar field to scale-resolved MPS matrices
        A_vec = A.reshape((1, -1))  # Flatten function
        
        # Reshape into scale resolving representation B
        w = '0'*2*self.n_bits                                 # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
        B_vec = np.zeros(4**self.n_bits).reshape((1, -1))     # similar to F but with scale indices

        for _ in range(4**self.n_bits):
            A_index = self.get_A_index(w)                # get original index for w
            B_index = int(w, 2)                     # get corresponding index for w
            w = bin(B_index+1)[2:].zfill(2*self.n_bits)       # set w += 1 in binary
            B_vec[0, B_index] = A_vec[0, A_index]   

        node = B_vec    # set first node of MPS
        MPS = []        # create MPS as list of matrices

        for _ in range(self.n_bits-1):
            m, n = node.shape
            node = node.reshape((4*m, int(n/4)))
            U, S, V = self.svd(node, chi)        # svd
            MPS.append(U)                   # save U as first node of MPS
            node = np.matmul(S, V)          # create remaining matrix S*V for next expansion step

        m, n = node.shape
        node = node.reshape((4*m, int(n/4)))
        MPS.append(node)    # add last node to MPS

        return MPS


    def convert_to_VF2D(self, MPS):   
        # converts scale-resolved MPS matrices to scalar field
        node_L = MPS[0]
        for i in range(1, self.n_bits):
            m, n = node_L.shape
            node_R = MPS[i].reshape((n, -1))
            node_L = np.matmul(node_L, node_R)
            m, n = node_L.shape
            node_L = node_L.reshape((4*m, int(n/4)))
        B_vec = node_L.reshape((1, -1)) 

        w = '0'*2*self.n_bits                            # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
        A_vec = np.zeros(4**self.n_bits).reshape((1, -1))     # similar to B but with dimensional indices

        for _ in range(4**self.n_bits):
            A_index = self.get_A_index(w)             
            B_index = int(w, 2)                   
            w = bin(B_index+1)[2:].zfill(2*self.n_bits)     
            A_vec[0, A_index]  = B_vec[0, B_index]

        return A_vec.reshape((self.N, self.N))


    def convert_MPS_to_cupy(self, tensor_list, dim_p):
        arrays = []
        for tensor in tensor_list:
            m, n = tensor.shape
            dim_left_bond = int(m/dim_p)
            dim_right_bond = n
            data = tensor.reshape((dim_left_bond, dim_p, dim_right_bond))
            arrays.append(cp.asarray(data))
        
        return arrays


    def convert_cupy_to_MPS(self, mps):
        arrays = []
        for tensor in mps:
            l, p, r = tensor.shape
            arrays.append(tensor.reshape((l*p, r)))

        return arrays


    def convert_ls(self, A):              
        # converts normal scalar field to scale-resolved scalar field
        A_vec = A.reshape((1, -1))  # Flatten function
        
        # Reshape into scale resolving representation C
        w = '0'*2*self.n_bits                                 # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
        B_vec = np.zeros(4**self.n_bits).reshape((1, -1))     # similar to A but with scale indices

        for _ in range(4**self.n_bits):
            A_index = self.get_A_index(w)                # get original index for w
            B_index = int(w, 2)                     # get corresponding index for w
            w = bin(B_index+1)[2:].zfill(2*self.n_bits)       # set w += 1 in binary
            B_vec[0, B_index] = A_vec[0, A_index]   

        return B_vec.reshape((self.N, self.N))


    def convert_back(self,A):            
        # converts scale-resolved scalar field to normal scalar field
        A_vec = A.reshape((1, -1))  # Flatten function
        
        # Reshape into scale resolving representation C
        w = '0'*2*self.n_bits                                 # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
        B_vec = np.zeros(4**self.n_bits).reshape((1, -1))     # similar to A but with scale indices

        for _ in range(4**self.n_bits):
            A_index = self.get_A_index(w)                # get original index for w
            B_index = int(w, 2)                     # get corresponding index for w
            w = bin(B_index+1)[2:].zfill(2*self.n_bits)       # set w += 1 in binary
            B_vec[0, A_index] = A_vec[0, B_index]   

        return B_vec.reshape((self.N, self.N))


    def hadamard_product_MPO(self, mps, options=None):
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
        
        k_delta = cp.zeros((4, 4, 4), dtype='float64')  # initialize kronecker delta as np.array
        for i in range(4):
            k_delta[i, i, i] = 1    # only set variables to one where each index is the same
        mpo = mps.copy()
        for i, tensor in enumerate(mpo):
            mpo[i] = contract('ijk, jlm->ilkm', tensor, k_delta, options=options)
        
        return mpo   # return the MPO


    def get_precontracted_LR_mps_mps(self, mps_2, mps_1, center=0, options=None):
        # prepare precontracted networks for dmrg sweeps
        # mps_1:    o--o-- center --o--o
        #           |  |            |  |
        # mps_2:    o--o-- center --o--o
        #           left            right networks

        left_networks = [None]*self.n_bits    # create a list containing the contracted left network for each site
        right_networks = [None]*self.n_bits   # create a list containing the contracted right network for each site

        # handle boundary networks
        dummy_t = cp.ones((1, 1))   # create a dummy network consisting of a 1
        left_networks[0] = dummy_t
        right_networks[-1] = dummy_t

        o = OptimizerOptions(path=[(0, 2), (0, 1)])

        if 'g_p_L_mps_mps' not in self.networks.keys():
            self.networks['g_p_L_mps_mps'] = [None]*self.n_bits
            self.networks['g_p_R_mps_mps'] = [None]*self.n_bits

        # from left to right
        for i in range(center):
            A = mps_1[i]
            B = mps_2[i]
            F = left_networks[i]

            if self.networks['g_p_L_mps_mps'][i] is None:
                self.networks['g_p_L_mps_mps'][i] = Network('apb, cpd, ac->bd', A, B, F, options=options)
                path, info = self.networks['g_p_L_mps_mps'][i].contract_path(optimize=o)
            else:
                self.networks['g_p_L_mps_mps'][i].reset_operands(A, B, F)

            F_new = self.networks['g_p_L_mps_mps'][i].contract()
            self.networks['g_p_L_mps_mps'][i].workspace_ptr = None
            # F_new = contract('apb, cpd, ac->bd', A, B, F, options=options, optimize=o)
            
            left_networks[i+1] = F_new
        
        # from right to left
        for i in range(self.n_bits-1, center, -1):
            A = mps_1[i]
            B = mps_2[i]
            F = right_networks[i]

            if self.networks['g_p_R_mps_mps'][i] is None:
                self.networks['g_p_R_mps_mps'][i] = Network('apb, cpd, bd->ac', A, B, F, options=options)
                path, info = self.networks['g_p_R_mps_mps'][i].contract_path(optimize=o)
            else:
                self.networks['g_p_R_mps_mps'][i].reset_operands(A, B, F)

            F_new = self.networks['g_p_R_mps_mps'][i].contract()
            self.networks['g_p_R_mps_mps'][i].workspace_ptr = None
            # F_new = contract('apb, cpd, bd->ac', A, B, F, options=options, optimize=o)
            
            right_networks[i-1] = F_new

        return left_networks, right_networks


    def get_precontracted_LR_mps_mpo(self, mps_2, mpo, mps_1, center=0, extra='', options=None):
        # prepare precontracted networks for dmrg sweeps
        # mps_1:    o--o-- center --o--o
        #           |  |            |  |
        # mpo:      0--0--        --0--0
        #           |  |            |  |
        # mps_2:    o--o-- center --o--o
        #           left            right networks

        left_networks = [None]*self.n_bits    # create a list containing the contracted left network for each site
        right_networks = [None]*self.n_bits   # create a list containing the contracted right network for each site

        # handle boundary networks
        dummy_t = cp.ones((1, 1, 1))   # create a dummy network consisting of a 1
        left_networks[0] = dummy_t
        right_networks[-1] = dummy_t

        o = OptimizerOptions(path=[(0, 3), (0, 2), (0, 1)])
        
        if f'g_p_L_mps_mpo{extra}' not in self.networks.keys():
            self.networks[f'g_p_L_mps_mpo{extra}'] = [None]*self.n_bits
            self.networks[f'g_p_R_mps_mpo{extra}'] = [None]*self.n_bits

        # from left to right
        for i in range(center):
            A = mps_1[i]
            B = mps_2[i]
            W = mpo[i]
            F = left_networks[i]

            if self.networks[f'g_p_L_mps_mpo{extra}'][i] is None:
                self.networks[f'g_p_L_mps_mpo{extra}'][i] = Network('apb, lprP, cPd, alc->brd', A, W, B, F, options=options)
                path, info = self.networks[f'g_p_L_mps_mpo{extra}'][i].contract_path(optimize=o)
            else:
                self.networks[f'g_p_L_mps_mpo{extra}'][i].reset_operands(A, W, B, F)

            F_new = self.networks[f'g_p_L_mps_mpo{extra}'][i].contract()
            self.networks[f'g_p_L_mps_mpo{extra}'][i].workspace_ptr = None
            # F_new = contract('apb, lprP, cPd, alc->brd', A, W, B, F, options=options, optimize=o)
            
            left_networks[i+1] = F_new

        # from right to left
        for i in range(self.n_bits-1, center, -1):
            A = mps_1[i]
            B = mps_2[i]
            W = mpo[i]
            F = right_networks[i]

            if self.networks[f'g_p_R_mps_mpo{extra}'][i] is None:
                self.networks[f'g_p_R_mps_mpo{extra}'][i] = Network('apb, lprP, cPd, brd->alc', A, W, B, F, options=options)
                path, info = self.networks[f'g_p_R_mps_mpo{extra}'][i].contract_path(optimize=o)
            else:
                self.networks[f'g_p_R_mps_mpo{extra}'][i].reset_operands(A, W, B, F)

            F_new = self.networks[f'g_p_R_mps_mpo{extra}'][i].contract()
            self.networks[f'g_p_R_mps_mpo{extra}'][i].workspace_ptr = None
            # F_new = contract('apb, lprP, cPd, brd->alc', A, W, B, F, options=options, optimize=o)
            
            right_networks[i-1] = F_new

        return left_networks, right_networks


    def update_precontracted_LR_mps_mps(self, F, B, A, LR, pos, extra='', options=None):
        # update the precontracted networks for dmrg sweeps
        #                        F--A--
        # For LR='L' contract :  F  |
        #                        F--B--
        #
        #                        --A--F
        # For LR='R' contract :    |  F
        #                        --B--F

        operands = [A, B, F]
        o = OptimizerOptions(path=[(0, 2), (0, 1)])

        if f'u_p_{LR}_mps_mps{extra}' not in self.networks.keys():
            self.networks[f'u_p_{LR}_mps_mps{extra}'] = [None]*self.n_bits

        if self.networks[f'u_p_{LR}_mps_mps{extra}'][pos] is None:
            if LR == 'L':
                self.networks[f'u_p_L_mps_mps{extra}'][pos] = Network('apb, cpd, ac->bd', *operands, options=options)
            elif LR == 'R':
                self.networks[f'u_p_R_mps_mps{extra}'][pos] = Network('apb, cpd, bd->ac', *operands, options=options)
            path, info = self.networks[f'u_p_{LR}_mps_mps{extra}'][pos].contract_path(optimize = o)
        else:
            self.networks[f'u_p_{LR}_mps_mps{extra}'][pos].reset_operands(*operands)
        
        F_new = self.networks[f'u_p_{LR}_mps_mps{extra}'][pos].contract()
        self.networks[f'u_p_{LR}_mps_mps{extra}'][pos].workspace_ptr = None
        
        return F_new


    def update_precontracted_LR_mps_mpo(self, F, B, W, A, LR, pos, extra='', options=None):
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

        operands = [A, W, B, F]
        o = OptimizerOptions(path=[(0, 3), (0, 2), (0, 1)])

        if f'u_p_{LR}_mps_mpo{extra}' not in self.networks.keys():
            self.networks[f'u_p_{LR}_mps_mpo{extra}'] = [None]*self.n_bits

        if self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos] is None:
            if LR == 'L':
                self.networks[f'u_p_L_mps_mpo{extra}'][pos] = Network('apb, lprP, cPd, alc->brd', *operands, options=options)
            elif LR == 'R':
                self.networks[f'u_p_R_mps_mpo{extra}'][pos] = Network('apb, lprP, cPd, brd->alc', *operands, options=options)
            path, info = self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos].contract_path(optimize = o)
        else:
            self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos].reset_operands(*operands)
        
        F_new = self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos].contract()
        self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos].workspace_ptr = None
        
        return F_new


    def Ax(self, H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, pos, options=None):
        # gives tensors corresponding to A*x
        # A = (1 - H_11, -H_12)
        #     (-H_21, 1 - H_22)
        Ax_1 = x_1.copy()
        Ax_2 = x_2.copy()
        operands_11 = [H_11_left, H_11_right, x_1, d1x_d1x]
        operands_12 = [H_12_left, H_12_right, x_2, d1x_d1y]
        operands_21 = [H_12_left, H_12_right, x_1, d1x_d1y]
        operands_22 = [H_22_left, H_22_right, x_2, d1y_d1y]

        o = OptimizerOptions(path=[(0, 2), (1, 2), (0, 1)])


        if 'cg_d' not in self.networks.keys():
            self.networks['cg_d'] = [None]*self.n_bits
            self.networks['cg_12'] = [None]*self.n_bits
            self.networks['cg_21'] = [None]*self.n_bits

        if self.networks[f'cg_d'][pos] is None:
            self.networks['cg_d'][pos] = Network('umd, reD, upr, mpeP->dPD', *operands_11, options=options)
            path, info = self.networks['cg_d'][pos].contract_path(optimize=o)
            self.networks['cg_12'][pos] = Network('umd, reD, upr, mpeP->dPD', *operands_12, options=options) 
            path, info = self.networks['cg_12'][pos].contract_path(optimize=o)
            self.networks['cg_21'][pos] = Network('umd, reD, dPD, mpeP->upr', *operands_21, options=options) 
            path, info = self.networks['cg_21'][pos].contract_path(optimize=o)
        else:
            self.networks['cg_d'][pos].reset_operands(*operands_11)
            self.networks['cg_12'][pos].reset_operands(*operands_12)
            self.networks['cg_21'][pos].reset_operands(*operands_21)
        
        Ax_1 -= self.networks['cg_d'][pos].contract() + self.networks['cg_12'][pos].contract()
        Ax_2 -= self.networks['cg_21'][pos].contract()

        self.networks['cg_d'][pos].reset_operands(*operands_22)
        Ax_2 -= self.networks['cg_d'][pos].contract()

        self.networks['cg_d'][pos].workspace_ptr = None
        self.networks['cg_12'][pos].workspace_ptr = None
        self.networks['cg_21'][pos].workspace_ptr = None

        return Ax_1, Ax_2


    # conjugate gradient algorithm in MPS form
    def solve_LS_cg(self, H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, b_1, b_2, pos, options=None):
        Ax_1, Ax_2 = self.Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, pos, options)
        r_1 = b_1 - Ax_1
        r_2 = b_2 - Ax_2
        p_1 = r_1
        p_2 = r_2

        o = OptimizerOptions(path=[(0, 1)])
        if 'residual' not in self.networks.keys():
            self.networks['residual'] = [None]*self.n_bits

        if self.networks['residual'][pos] is None:
            self.networks['residual'][pos] = Network('apb, apb->', r_1, r_1, options=options) 
            path, info = self.networks['residual'][pos].contract_path(optimize=o)
        else:
            self.networks['residual'][pos].reset_operands(r_1, r_1)
        r_r = self.networks['residual'][pos].contract()
        self.networks['residual'][pos].reset_operands(r_2, r_2)
        r_r += self.networks['residual'][pos].contract()

        iter = 0
        # n = 2
        # for s in b_1.shape:
        #     n *= s
        # max_iter = 10*n
        max_iter = 100
        while r_r > 1e-5 and iter < max_iter:
            iter += 1

            Ap_1, Ap_2 = self.Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, p_1, p_2, d1x_d1x, d1x_d1y, d1y_d1y, pos, options)
            self.networks['residual'][pos].reset_operands(p_1, Ap_1)
            pAp_1 = self.networks['residual'][pos].contract()
            self.networks['residual'][pos].reset_operands(p_2, Ap_2)
            pAp_2 = self.networks['residual'][pos].contract()
            alpha = r_r / (pAp_1 + pAp_2)

            x_1 = x_1 + alpha * p_1
            x_2 = x_2 + alpha * p_2

            Ax_1, Ax_2 = self.Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, pos, options)
            r_new_1 = b_1 - Ax_1
            r_new_2 = b_2 - Ax_2
            
            self.networks['residual'][pos].reset_operands(r_new_1, r_new_1)
            r_new_r_new = self.networks['residual'][pos].contract()
            self.networks['residual'][pos].reset_operands(r_new_2, r_new_2)
            r_new_r_new += self.networks['residual'][pos].contract()
            beta = r_new_r_new / r_r

            p_1 = r_new_1 + beta * p_1
            p_2 = r_new_2 + beta * p_2

            r_r = r_new_r_new
        
        # print(iter, r_r)
        self.networks['residual'][pos].workspace_ptr = None

        if self.meas_cg_iter:
            self.cg_iter_data = (iter, float(r_r))

        return x_1, x_2


    # linear system solver via matrix inversion in MPS form
    def solve_LS_inv(self, H_11, H_12, H_22, b_1, b_2):
        shape = b_1.shape
        dim = 1
        for d in shape:
            dim *= d
        b_1 = b_1.flatten().get()
        b_2 = b_2.flatten().get()
        H_11 = H_11.reshape((dim, dim)).get()
        H_12 = H_12.reshape((dim, dim)).get()
        H_22 = H_22.reshape((dim, dim)).get()
        
        H = np.block([[H_11, H_12.T], [H_12, H_22]])
        A = np.eye(len(H)) - H
        b = np.concatenate((b_1, b_2))
        
        x = np.linalg.solve(A,b)
        U_new, V_new = np.array_split(x, 2)

        return cp.asarray(U_new.reshape(shape)), cp.asarray(V_new.reshape(shape))


    # linear system solver via scipy.cg in MPS form
    def solve_LS_cg_scipy(self, H_11, H_12, H_22, x_1, x_2, b_1, b_2):
        shape = x_1.shape
        dim = 1
        for d in shape:
            dim *= d
        b_1 = b_1.flatten().get()
        b_2 = b_2.flatten().get()
        x_1 = x_1.flatten().get()
        x_2 = x_2.flatten().get()
        H_11 = H_11.reshape((dim, dim)).get()
        H_12 = H_12.reshape((dim, dim)).get()
        H_22 = H_22.reshape((dim, dim)).get()
        
        H = np.block([[H_11, H_12.T], [H_12, H_22]])
        A = np.eye(len(H)) - H
        b = np.concatenate((b_1, b_2))
        x = np.concatenate((x_1, x_2))
        # print(np.linalg.cond(A))
        x_sol = cg(A, b, x)
        U_new, V_new = np.array_split(x_sol[0], 2)

        return cp.asarray(U_new.reshape(shape)), cp.asarray(V_new.reshape(shape))


    # helper function to compute convection-diffusion terms
    def left_right_A_W(self, left_tn, right_tn, A_t, W_t, pos, extra='', options=None, contract_string='umd, upr, mpeP, reD->dPD'):
        operands = [left_tn, A_t, W_t, right_tn]
        o = OptimizerOptions(path=[(0, 1), (0, 2), (0, 1)])

        if f'l_r_A_W{extra}' not in self.networks.keys():
            self.networks[f'l_r_A_W{extra}'] = [None]*self.n_bits

        if self.networks[f'l_r_A_W{extra}'][pos] is None:
            self.networks[f'l_r_A_W{extra}'][pos] = Network(contract_string, *operands, options=options)
            path, info = self.networks[f'l_r_A_W{extra}'][pos].contract_path(optimize = o)
        else:
            self.networks[f'l_r_A_W{extra}'][pos].reset_operands(*operands)

        tensor = self.networks[f'l_r_A_W{extra}'][pos].contract()
        self.networks[f'l_r_A_W{extra}'][pos].workspace_ptr = None

        return tensor


    def left_right_W(self, left_tn, right_tn, W_t, options=None, contract_string='umd, mpeP, reD->uprdPD'):
        o = OptimizerOptions(path=[(0, 1), (0, 1)])

        return contract(contract_string, left_tn, W_t, right_tn, options=options, optimize=o)


    # time stepping function
    def single_time_step(self, dt, U, V, Ax_MPS, Ay_MPS, Bx_MPS, By_MPS, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver='cg', options=None):
        
        # initialize precontracted left and right networks
        U_Ax_left, U_Ax_right = self.get_precontracted_LR_mps_mps(U, Ax_MPS, 0, options)
        V_Ay_left, V_Ay_right = self.get_precontracted_LR_mps_mps(V, Ay_MPS, 0, options)

        # create MPOs for convection-diffusion terms
        mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'V', 'max_extent': self.chi_mpo}}
        
        Bx_MPO = self.hadamard_product_MPO(Bx_MPS, options)
        Bxd1x = self.multiply_mpo_mpo(Bx_MPO, d1x, mult_algorithm, options)
        d1xBx = self.multiply_mpo_mpo(d1x, Bx_MPO, mult_algorithm, options)
        By_MPO = self.hadamard_product_MPO(By_MPS, options)
        Byd1y = self.multiply_mpo_mpo(By_MPO, d1y, mult_algorithm, options)
        d1yBy = self.multiply_mpo_mpo(d1y, By_MPO, mult_algorithm, options)

        # convection-diffusion terms for x direction (prefactors not included)
        U_d2x_Bx_left, U_d2x_Bx_right = self.get_precontracted_LR_mps_mpo(U, d2x, Bx_MPS, 0, '_d', options)
        U_d2y_Bx_left, U_d2y_Bx_right = self.get_precontracted_LR_mps_mpo(U, d2y, Bx_MPS, 0, '_d', options)
        U_Bxd1x_Bx_left, U_Bxd1x_Bx_right = self.get_precontracted_LR_mps_mpo(U, Bxd1x, Bx_MPS, 0, '_f', options)
        U_d1xBx_Bx_left, U_d1xBx_Bx_right = self.get_precontracted_LR_mps_mpo(U, d1xBx, Bx_MPS, 0, '_f', options)
        U_Byd1y_Bx_left, U_Byd1y_Bx_right = self.get_precontracted_LR_mps_mpo(U, Byd1y, Bx_MPS, 0, '_f', options)
        U_d1yBy_Bx_left, U_d1yBy_Bx_right = self.get_precontracted_LR_mps_mpo(U, d1yBy, Bx_MPS, 0, '_f', options)

        # convection-diffusion terms for y direction (prefactors not included)
        V_d2x_By_left, V_d2x_By_right = self.get_precontracted_LR_mps_mpo(V, d2x, By_MPS, 0, '_d', options)
        V_d2y_By_left, V_d2y_By_right = self.get_precontracted_LR_mps_mpo(V, d2y, By_MPS, 0, '_d', options)
        V_Bxd1x_By_left, V_Bxd1x_By_right = self.get_precontracted_LR_mps_mpo(V, Bxd1x, By_MPS, 0, '_f', options)
        V_d1xBx_By_left, V_d1xBx_By_right = self.get_precontracted_LR_mps_mpo(V, d1xBx, By_MPS, 0, '_f', options)
        V_Byd1y_By_left, V_Byd1y_By_right = self.get_precontracted_LR_mps_mpo(V, Byd1y, By_MPS, 0, '_f', options)
        V_d1yBy_By_left, V_d1yBy_By_right = self.get_precontracted_LR_mps_mpo(V, d1yBy, By_MPS, 0, '_f', options)

        epsilon = 1e-5              # convergence criterion
        E_0 = 1e-10                 # initialize energy before
        # E_1 = 2*epsilon             # initialize energy after
        operands = [U[0], U[0]]
        o = OptimizerOptions(path=[(0, 1)])
        if 'norm' not in self.networks.keys():
            self.networks['norm'] = Network('apb, apb->', *operands, options=options)
            path, info = self.networks['norm'].contract_path(optimize = o)
        else:
            self.networks['norm'].reset_operands(*operands)

        E_1_U = self.networks['norm'].contract()
        operands = [V[0], V[0]]
        self.networks['norm'].reset_operands(*operands)
        E_1_V = self.networks['norm'].contract()
        self.networks['norm'].workspace_ptr = None
        E_1 = E_1_U + E_1_V
        
        run = 0
        while np.abs((E_1-E_0)/E_0) > epsilon:      # do until the state does not change anymore
            run += 1

            # sweep through MPS and optimize locally
            for i in range(self.n_bits-1):      
                # Build linear system Ax = b 
                # b_1
                operands = [U_Ax_left[i], Ax_MPS[i], U_Ax_right[i]]
                o = OptimizerOptions(path=[(0, 1), (0, 1)])

                if 'b' not in self.networks.keys():
                    self.networks['b'] = [None]*self.n_bits

                if self.networks['b'][i] is None:
                    self.networks['b'][i] = Network('ud, upr, rD->dpD', *operands, options=options)
                    path, info = self.networks['b'][i].contract_path(optimize = o)
                else:
                    self.networks['b'][i].reset_operands(*operands)

                b_1 = self.networks['b'][i].contract()
                # b_1 = contract('ud, upr, rD->dpD', U_Ax_left[i], Ax_MPS[i], U_Ax_right[i], options=options)

                # convection-diffusion terms
                b_1 += dt/self.Re * self.left_right_A_W(U_d2x_Bx_left[i], U_d2x_Bx_right[i], Bx_MPS[i], d2x[i], i, '_d', options)
                b_1 += dt/self.Re * self.left_right_A_W(U_d2y_Bx_left[i], U_d2y_Bx_right[i], Bx_MPS[i], d2y[i], i, '_d', options)
                b_1 += -dt/2 * self.left_right_A_W(U_Bxd1x_Bx_left[i], U_Bxd1x_Bx_right[i], Bx_MPS[i], Bxd1x[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_d1xBx_Bx_left[i], U_d1xBx_Bx_right[i], Bx_MPS[i], d1xBx[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_Byd1y_Bx_left[i], U_Byd1y_Bx_right[i], Bx_MPS[i], Byd1y[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_d1yBy_Bx_left[i], U_d1yBy_Bx_right[i], Bx_MPS[i], d1yBy[i], i, '_f', options)
                
                # b_2
                operands = [V_Ay_left[i], Ay_MPS[i], V_Ay_right[i]]
                self.networks['b'][i].reset_operands(*operands)
                b_2 = self.networks['b'][i].contract()
                self.networks['b'][i].workspace_ptr = None
                # b_2 = contract('ud, upr, rD->dpD', V_Ay_left[i], Ay_MPS[i], V_Ay_right[i], options=options)

                # convection-diffusion terms
                b_2 += dt/self.Re * self.left_right_A_W(V_d2x_By_left[i], V_d2x_By_right[i], By_MPS[i], d2x[i], i, '_d', options)
                b_2 += dt/self.Re * self.left_right_A_W(V_d2y_By_left[i], V_d2y_By_right[i], By_MPS[i], d2y[i], i, '_d', options)
                b_2 += -dt/2 * self.left_right_A_W(V_Bxd1x_By_left[i], V_Bxd1x_By_right[i], By_MPS[i], Bxd1x[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_d1xBx_By_left[i], V_d1xBx_By_right[i], By_MPS[i], d1xBx[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_Byd1y_By_left[i], V_Byd1y_By_right[i], By_MPS[i], Byd1y[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_d1yBy_By_left[i], V_d1yBy_By_right[i], By_MPS[i], d1yBy[i], i, '_f', options)

                # solve linear system
                if solver == 'cg':
                    U_new, V_new = self.solve_LS_cg(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], U[i], V[i], self.mu*dt**2 * d1x_d1x[i], self.mu*dt**2 * d1x_d1y[i], self.mu*dt**2 * d1y_d1y[i], b_1, b_2, i, options)
                elif solver == 'inv':
                    H_11 = self.mu*dt**2 * self.left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i], options)
                    H_12 = self.mu*dt**2 * self.left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i], options)
                    H_22 = self.mu*dt**2 * self.left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i], options)

                    U_new, V_new = self.solve_LS_inv(H_11, H_12, H_22, b_1, b_2)
                elif solver == 'scipy.cg':
                    H_11 = self.mu*dt**2 * self.left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i], options)
                    H_12 = self.mu*dt**2 * self.left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i], options)
                    H_22 = self.mu*dt**2 * self.left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i], options)
                    x_1 = U[i].copy()
                    x_2 = V[i].copy()

                    U_new, V_new = self.solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2)
                else:
                    raise Exception(f"The solver '{solver}' is not known. Please use 'cg', 'inv', or 'scipy.cg' instead.")

                # update MPSs and precontracted networks
                # update MPS
                U[i] = U_new
                V[i] = V_new

                # shift canonical center 
                U = self.shift_canonical_center(U, i+1, i, options)
                V = self.shift_canonical_center(V, i+1, i, options)

                # update precontracted networks
                U_Ax_left[i+1] = self.update_precontracted_LR_mps_mps(U_Ax_left[i], U[i], Ax_MPS[i], 'L', i, options)
                V_Ay_left[i+1] = self.update_precontracted_LR_mps_mps(V_Ay_left[i], V[i], Ay_MPS[i], 'L', i, options)

                U_d2x_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d2x_Bx_left[i], U[i], d2x[i], Bx_MPS[i], 'L', i, '_d', options)
                U_d2y_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d2y_Bx_left[i], U[i], d2y[i], Bx_MPS[i], 'L', i, '_d', options)
                U_Bxd1x_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_Bxd1x_Bx_left[i], U[i], Bxd1x[i], Bx_MPS[i], 'L', i, '_f', options)
                U_d1xBx_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d1xBx_Bx_left[i], U[i], d1xBx[i], Bx_MPS[i], 'L', i, '_f', options)
                U_Byd1y_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_Byd1y_Bx_left[i], U[i], Byd1y[i], Bx_MPS[i], 'L', i, '_f', options)
                U_d1yBy_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d1yBy_Bx_left[i], U[i], d1yBy[i], Bx_MPS[i], 'L', i, '_f', options)

                V_d2x_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d2x_By_left[i], V[i], d2x[i], By_MPS[i], 'L', i, '_d', options)
                V_d2y_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d2y_By_left[i], V[i], d2y[i], By_MPS[i], 'L', i, '_d', options)
                V_Bxd1x_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_Bxd1x_By_left[i], V[i], Bxd1x[i], By_MPS[i], 'L', i, '_f', options)
                V_d1xBx_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d1xBx_By_left[i], V[i], d1xBx[i], By_MPS[i], 'L', i, '_f', options)
                V_Byd1y_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_Byd1y_By_left[i], V[i], Byd1y[i], By_MPS[i], 'L', i, '_f', options)
                V_d1yBy_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d1yBy_By_left[i], V[i], d1yBy[i], By_MPS[i], 'L', i, '_f', options)

                U_d1x_d1x_U_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d1x_d1x_U_left[i], U[i], d1x_d1x[i], U[i], 'L', i, '_dd', options)
                U_d1x_d1y_V_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d1x_d1y_V_left[i], U[i], d1x_d1y[i], V[i], 'L', i, '_ddxy', options)
                V_d1y_d1y_V_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d1y_d1y_V_left[i], V[i], d1y_d1y[i], V[i], 'L', i, '_dd', options)
            
            # sweep back through MPS and optimize locally
            for i in range(self.n_bits-1, 0, -1):
                # Build linear system Ax = b 
                # b_1
                operands = [U_Ax_left[i], Ax_MPS[i], U_Ax_right[i]]
                if self.networks['b'][i] is None:
                    self.networks['b'][i] = Network('ud, upr, rD->dpD', *operands, options=options)
                    path, info = self.networks['b'][i].contract_path(optimize = o)
                else:
                    self.networks['b'][i].reset_operands(*operands)
                b_1 = self.networks['b'][i].contract()
                # b_1 = contract('ud, upr, rD->dpD', U_Ax_left[i], Ax_MPS[i], U_Ax_right[i], options=options)

                # convection-diffusion terms
                b_1 += dt/self.Re * self.left_right_A_W(U_d2x_Bx_left[i], U_d2x_Bx_right[i], Bx_MPS[i], d2x[i], i, '_d', options)
                b_1 += dt/self.Re * self.left_right_A_W(U_d2y_Bx_left[i], U_d2y_Bx_right[i], Bx_MPS[i], d2y[i], i, '_d', options)
                b_1 += -dt/2 * self.left_right_A_W(U_Bxd1x_Bx_left[i], U_Bxd1x_Bx_right[i], Bx_MPS[i], Bxd1x[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_d1xBx_Bx_left[i], U_d1xBx_Bx_right[i], Bx_MPS[i], d1xBx[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_Byd1y_Bx_left[i], U_Byd1y_Bx_right[i], Bx_MPS[i], Byd1y[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_d1yBy_Bx_left[i], U_d1yBy_Bx_right[i], Bx_MPS[i], d1yBy[i], i, '_f', options)

                # b_2
                operands = [V_Ay_left[i], Ay_MPS[i], V_Ay_right[i]]
                self.networks['b'][i].reset_operands(*operands)
                b_2 = self.networks['b'][i].contract()
                self.networks['b'][i].workspace_ptr = None
                # b_2 = contract('ud, upr, rD->dpD', V_Ay_left[i], Ay_MPS[i], V_Ay_right[i], options=options)

                # convection-diffusion terms
                b_2 += dt/self.Re * self.left_right_A_W(V_d2x_By_left[i], V_d2x_By_right[i], By_MPS[i], d2x[i], i, '_d', options)
                b_2 += dt/self.Re * self.left_right_A_W(V_d2y_By_left[i], V_d2y_By_right[i], By_MPS[i], d2y[i], i, '_d', options)
                b_2 += -dt/2 * self.left_right_A_W(V_Bxd1x_By_left[i], V_Bxd1x_By_right[i], By_MPS[i], Bxd1x[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_d1xBx_By_left[i], V_d1xBx_By_right[i], By_MPS[i], d1xBx[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_Byd1y_By_left[i], V_Byd1y_By_right[i], By_MPS[i], Byd1y[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_d1yBy_By_left[i], V_d1yBy_By_right[i], By_MPS[i], d1yBy[i], i, '_f', options)

                # solve linear system
                if solver == 'cg':
                    U_new, V_new = self.solve_LS_cg(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], U[i], V[i], self.mu*dt**2 * d1x_d1x[i], self.mu*dt**2 * d1x_d1y[i], self.mu*dt**2 * d1y_d1y[i], b_1, b_2, i, options)
                elif solver == 'inv':
                    H_11 = self.mu*dt**2 * self.left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i], options)
                    H_12 = self.mu*dt**2 * self.left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i], options)
                    H_22 = self.mu*dt**2 * self.left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i], options)

                    U_new, V_new = self.solve_LS_inv(H_11, H_12, H_22, b_1, b_2)
                elif solver == 'scipy.cg':
                    H_11 = self.mu*dt**2 * self.left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i], options)
                    H_12 = self.mu*dt**2 * self.left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i], options)
                    H_22 = self.mu*dt**2 * self.left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i], options)
                    x_1 = U[i].copy()
                    x_2 = V[i].copy()

                    U_new, V_new = self.solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2)
                else:
                    raise Exception(f"The solver '{solver}' is not known. Please use 'cg', 'inv', or 'scipy.cg' instead.")

                # update MPSs and precontracted networks
                # update MPS
                U[i] = U_new
                V[i] = V_new

                # shift canonical center 
                U = self.shift_canonical_center(U, i-1, i, options)
                V = self.shift_canonical_center(V, i-1, i, options)

                # update precontracted networks
                U_Ax_right[i-1] = self.update_precontracted_LR_mps_mps(U_Ax_right[i], U[i], Ax_MPS[i], 'R', i, options)
                V_Ay_right[i-1] = self.update_precontracted_LR_mps_mps(V_Ay_right[i], V[i], Ay_MPS[i], 'R', i, options)

                U_d2x_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d2x_Bx_right[i], U[i], d2x[i], Bx_MPS[i], 'R', i, '_d', options)
                U_d2y_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d2y_Bx_right[i], U[i], d2y[i], Bx_MPS[i], 'R', i, '_d', options)
                U_Bxd1x_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_Bxd1x_Bx_right[i], U[i], Bxd1x[i], Bx_MPS[i], 'R', i, '_f', options)
                U_d1xBx_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d1xBx_Bx_right[i], U[i], d1xBx[i], Bx_MPS[i], 'R', i, '_f', options)
                U_Byd1y_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_Byd1y_Bx_right[i], U[i], Byd1y[i], Bx_MPS[i], 'R', i, '_f', options)
                U_d1yBy_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d1yBy_Bx_right[i], U[i], d1yBy[i], Bx_MPS[i], 'R', i, '_f', options)

                V_d2x_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d2x_By_right[i], V[i], d2x[i], By_MPS[i], 'R', i, '_d', options)
                V_d2y_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d2y_By_right[i], V[i], d2y[i], By_MPS[i], 'R', i, '_d', options)
                V_Bxd1x_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_Bxd1x_By_right[i], V[i], Bxd1x[i], By_MPS[i], 'R', i, '_f', options)
                V_d1xBx_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d1xBx_By_right[i], V[i], d1xBx[i], By_MPS[i], 'R', i, '_f', options)
                V_Byd1y_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_Byd1y_By_right[i], V[i], Byd1y[i], By_MPS[i], 'R', i, '_f', options)
                V_d1yBy_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d1yBy_By_right[i], V[i], d1yBy[i], By_MPS[i], 'R', i, '_f', options)

                U_d1x_d1x_U_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d1x_d1x_U_right[i], U[i], d1x_d1x[i], U[i], 'R', i, '_dd', options)
                U_d1x_d1y_V_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d1x_d1y_V_right[i], U[i], d1x_d1y[i], V[i], 'R', i, '_ddxy', options)
                V_d1y_d1y_V_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d1y_d1y_V_right[i], V[i], d1y_d1y[i], V[i], 'R', i, '_dd', options)

            E_0 = E_1       
            # E_1 = contract('apb, apb->', U[0], U[0], options=options) + contract('apb, apb->', V[0], V[0], options=options) 

            operands = [U[0], U[0]]
            self.networks['norm'].reset_operands(*operands)

            # operands = [U[0], U[0]]
            # o = OptimizerOptions(path=[(0, 1)])
            # if 'norm' not in networks.keys():
            #     networks['norm'] = Network('apb, apb->', *operands, options=options)
            #     path, info = networks['norm'].contract_path(optimize = o)
            # else:
            #     networks['norm'].reset_operands(*operands)

            E_1_U = self.networks['norm'].contract()
            operands = [V[0], V[0]]
            self.networks['norm'].reset_operands(*operands)
            E_1_V = self.networks['norm'].contract()
            self.networks['norm'].workspace_ptr = None
            E_1 = E_1_U + E_1_V
            print(f"Run: {run}, Diff: {(E_1-E_0)/E_0}, E_0: {E_0}, E_1: {E_1}", end='\r')
        print(f"Run: {run}, Diff: {(E_1-E_0)/E_0}, E_0: {E_0}, E_1: {E_1}", end='\r')
        
        return U, V


    def plot(self, U, V, time=-1, full=False, save_path=None, show=False):
        # plot velocity field given as MPS
        u = self.convert_to_VF2D(self.convert_cupy_to_MPS(U))
        v = self.convert_to_VF2D(self.convert_cupy_to_MPS(V))  

        # Genaral parameters
        x = np.linspace(0, 1-self.dx, self.N)
        y = np.linspace(0, 1-self.dx, self.N)
        Y, X = np.meshgrid(y, x)
        n_s = 2**(self.n_bits-4)                  # Plot N/n_s number of arrows

        plt.figure()
        plt.contourf(X, Y, Dx(v, self.dx)-Dy(u, self.dx), 100, cmap="seismic")
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


    # Free all network objects
    def free_networks(self, n_dict):
        for key, el in n_dict.items():
            if isinstance(el, dict):
                self.free_networks(el)
            elif isinstance(el, list):
                for el_ in el:
                    if el_ is not None:
                        el_.free()
            else:
                el.free()


    # time evolution algorithm
    def time_evolution(self):
        n_steps = int(np.ceil(self.T/self.dt))    # time steps
        # finite difference operators with 8th order precision
        d1x = Diff_1_8_x_MPO(self.n_bits, self.dx, self.options)
        d1y = Diff_1_8_y_MPO(self.n_bits, self.dx, self.options)
        d2x = Diff_2_8_x_MPO(self.n_bits, self.dx, self.options)
        d2y = Diff_2_8_y_MPO(self.n_bits, self.dx, self.options)

        # finite difference operators with 2nd order precision 
        # d1x = Diff_1_2_x_MPO(n, dx, options)
        # d1y = Diff_1_2_y_MPO(n, dx, options)
        # d2x = Diff_2_2_x_MPO(n, dx, options)
        # d2y = Diff_2_2_y_MPO(n, dx, options)

        mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'V', 'max_extent': self.chi_mpo}} # 'rel_cutoff':1e-10, 
        d1x_d1x = self.multiply_mpo_mpo(d1x, d1x, mult_algorithm, self.options)
        d1x_d1y = self.multiply_mpo_mpo(d1x, d1y, mult_algorithm, self.options)
        d1y_d1y = self.multiply_mpo_mpo(d1y, d1y, mult_algorithm, self.options)
        
        # bring the orthogonality center to the first tensor
        U = self.canonical_center(self.U_init, 0, self.options)
        V = self.canonical_center(self.V_init, 0, self.options)

        # initialize precontracted left and right networks
        U_d1x_d1x_U_left, U_d1x_d1x_U_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1x, U, 0, '_dd', self.options)
        U_d1x_d1y_V_left, U_d1x_d1y_V_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1y, V, 0, '_ddxy', self.options)
        V_d1y_d1y_V_left, V_d1y_d1y_V_right = self.get_precontracted_LR_mps_mpo(V, d1y_d1y, V, 0, '_dd', self.options)

        t = 0
        comp_time = {}
        cg_iter_data = {}
        print("Simulation begins!")
        for step in range(n_steps):   # for every time step dt
            print(f"Step = {step} - Time = {t}", end='\n')
            if self.path is not None and step%(int(n_steps/self.save_number)) == 0:
                np.save(f"{self.path}/u_time_{round(t, 5)}.npy", np.array([el.get() for el in U], dtype=object))
                np.save(f"{self.path}/v_time_{round(t, 5)}.npy", np.array([el.get() for el in V], dtype=object))

            U_trial = U.copy()         # trial velocity state
            V_trial = V.copy()         # trial velocity state

            U_prev = U.copy()          # previous velocity state
            V_prev = V.copy()          # previous velocity state

            U_prev_copy = U.copy()          # previous velocity state
            V_prev_copy = V.copy()          # previous velocity state

            if self.meas_comp_time:
                start = time.time()
            # Midpoint RK-2 step
            U_mid, V_mid = self.single_time_step(self.dt/2, U_trial, V_trial, U_prev, V_prev, U_prev_copy, V_prev_copy, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, options=self.options)
            # Full RK-2 step
            print('')
            U, V = self.single_time_step(self.dt, U_trial, V_trial, U_prev, V_prev, U_mid, V_mid, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, options=self.options)
            # U, V = single_time_step(U_trial, V_trial, U_prev, V_prev, U_prev_copy, V_prev_copy, chi_mpo, dt, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver=solver, options=options)
            print('\n')
            if self.meas_comp_time:
                end = time.time()
                comp_time[t] = end-start

                with open(self.comp_time_path, "w") as outfile: 
                    json.dump(comp_time, outfile)
            if self.meas_cg_iter:
                cg_iter_data[t] = self.cg_iter_data

                with open(self.cg_iter_path, "w") as outfile: 
                    json.dump(cg_iter_data, outfile)
            
            t += self.dt
            
        
        # plot(U, V, time=t, save_path=f"{save_path}/final.png", show=False)
        # np.save(f"{save_path}/u_final.npy", np.array([el.get() for el in U], dtype=object))
        # np.save(f"{save_path}/v_final.npy", np.array([el.get() for el in V], dtype=object))
        
        self.free_networks(self.networks)
        self.networks.clear()


    # extract occupied gpu memory
    def get_gpu_memory(self, gpu_id=0):
        n_steps = int(np.ceil(self.T/self.dt))    # time steps
        # finite difference operators with 8th order precision
        d1x = Diff_1_8_x_MPO(self.n_bits, self.dx, self.options)
        d1y = Diff_1_8_y_MPO(self.n_bits, self.dx, self.options)
        d2x = Diff_2_8_x_MPO(self.n_bits, self.dx, self.options)
        d2y = Diff_2_8_y_MPO(self.n_bits, self.dx, self.options)

        # finite difference operators with 2nd order precision 
        # d1x = Diff_1_2_x_MPO(n, dx, options)
        # d1y = Diff_1_2_y_MPO(n, dx, options)
        # d2x = Diff_2_2_x_MPO(n, dx, options)
        # d2y = Diff_2_2_y_MPO(n, dx, options)

        mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'V', 'max_extent': self.chi_mpo}} # 'rel_cutoff':1e-10, 
        d1x_d1x = self.multiply_mpo_mpo(d1x, d1x, mult_algorithm, self.options)
        d1x_d1y = self.multiply_mpo_mpo(d1x, d1y, mult_algorithm, self.options)
        d1y_d1y = self.multiply_mpo_mpo(d1y, d1y, mult_algorithm, self.options)
        
        # bring the orthogonality center to the first tensor
        U = self.canonical_center(self.U_init, 0, self.options)
        V = self.canonical_center(self.V_init, 0, self.options)

        # initialize precontracted left and right networks
        U_d1x_d1x_U_left, U_d1x_d1x_U_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1x, U, 0, '_dd', self.options)
        U_d1x_d1y_V_left, U_d1x_d1y_V_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1y, V, 0, '_ddxy', self.options)
        V_d1y_d1y_V_left, V_d1y_d1y_V_right = self.get_precontracted_LR_mps_mpo(V, d1y_d1y, V, 0, '_dd', self.options)

        U_trial = U.copy()         # trial velocity state
        V_trial = V.copy()         # trial velocity state

        U_prev = U.copy()          # previous velocity state
        V_prev = V.copy()          # previous velocity state

        U_prev_copy = U.copy()          # previous velocity state
        V_prev_copy = V.copy()          # previous velocity state

        if self.meas_comp_time:
            start = time.time()
        # Midpoint RK-2 step
        U_mid, V_mid = self.single_time_step(self.dt/2, U_trial, V_trial, U_prev, V_prev, U_prev_copy, V_prev_copy, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, options=self.options)
        # Full RK-2 step
        print('')
        U, V = self.single_time_step(self.dt, U_trial, V_trial, U_prev, V_prev, U_mid, V_mid, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, options=self.options)
        # U, V = single_time_step(U_trial, V_trial, U_prev, V_prev, U_prev_copy, V_prev_copy, chi_mpo, dt, Re, mu, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver=solver, options=options)
        print('\n')

        gpu_mem = float(get_gpu_memory()[gpu_id])
        
        self.free_networks(self.networks)
        self.networks.clear()

        return gpu_mem
            

    def build_initial_fields(self, y_min=0.4, y_max=0.6, h=1/200, u_max=1):
        # Generate initial fields
        U, V = self.initial_fields(y_min, y_max, h, u_max) 

        # Rescale into non-dimensional units
        U = U/u_max
        V = V/u_max

        # Convert them to MPS form
        MPS_U = self.convert_to_MPS2D(U, self.chi)
        MPS_V = self.convert_to_MPS2D(V, self.chi)

        # Tranform into quimb MPS form
        MPS_U_cupy= self.convert_MPS_to_cupy(MPS_U, 4)
        MPS_V_cupy = self.convert_MPS_to_cupy(MPS_V, 4)

        self.U_init = MPS_U_cupy
        self.V_init = MPS_V_cupy

        print("Initialized Fields")

    
    def set_initial_fields(self, U, V, u_max=1):
        # Rescale into non-dimensional units
        U = U/u_max
        V = V/u_max

        # Convert them to MPS form
        MPS_U = self.convert_to_MPS2D(U, self.chi)
        MPS_V = self.convert_to_MPS2D(V, self.chi)

        # Tranform into quimb MPS form
        MPS_U_cupy= self.convert_MPS_to_cupy(MPS_U, 4)
        MPS_V_cupy = self.convert_MPS_to_cupy(MPS_V, 4)

        self.U_init = MPS_U_cupy
        self.V_init = MPS_V_cupy

        print("Initialized Fields")