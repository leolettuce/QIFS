import numpy as np
from scipy.linalg import svd as plain_svd
from scipy.linalg import qr, rq

def svd(mat, chi=None):
    min_dim = np.min(mat.shape)
    if chi == None or chi >= min_dim:   # plain svd
        chi = min_dim
    U, S, V = plain_svd(mat, full_matrices=False)
    U = U[:, :chi]
    S = S[:chi]
    V = V[:chi, :]
    S = np.diag(S)

    return U, S, V

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


def canonicalize_mps_tensors(a, b, absorb='right'):
    # Perform canonicalization of two MPS tensors.
    if absorb == 'right':
        i_s, p_s, j_s = a.shape
        a = a.reshape((i_s*p_s, j_s))
        a, S, V = svd(a)
        a = a.reshape((i_s, p_s, -1))
        b = np.einsum('xj,jpk->xpk', np.matmul(S, V), b) # combine b with r
    elif absorb == 'left':
        j_s, p_s, k_s = b.shape
        b = b.reshape((j_s, p_s*k_s))
        U, S, b = svd(b)
        b = b.reshape((-1, p_s, k_s))
        a = np.einsum('jx,ipj->ipx', np.matmul(U, S), a) # combine a with r 
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

def canonicalize_mpo_tensors(a, b, absorb='right'):
    # Perform canonicalization of two MPO tensors.
    if absorb == 'right':
        l_s, u_s, r_s, d_s = a.shape
        a = a.transpose((0, 1, 3, 2)).reshape((l_s*u_s*d_s, r_s))
        a, r = qr(a)
        a = a.reshape((l_s, u_s. d_s, -1)).transpose((0, 1, 3, 2))
        b = np.einsum('xl,lurd->xurd', r, b) # combine b with r
    elif absorb == 'left':
        l_s, u_s, r_s, d_s = b.shape
        b = b.reshape((l_s, u_s*r_s*d_s))
        r, b = rq(b)
        b = b.reshape((-1, u_s, r_s, d_s))
        a = np.einsum('rx,lurd->luxd', r, a) # combine a with r 
    else:
        raise ValueError(f"absorb must be either left or right")
    return a, b


def canonicalize_mpo_tensors(a, b, absorb='right'):
    # Perform canonicalization of two MPS tensors.
    if absorb == 'right':
        l_s, u_s, r_s, d_s = a.shape
        a = a.transpose((0, 1, 3, 2)).reshape((l_s*u_s*d_s, r_s))
        a, S, V = svd(a)
        a = a.reshape((l_s, u_s. d_s, -1)).transpose((0, 1, 3, 2))
        b = np.einsum('xl,lurd->xurd', np.matmul(S, V), b) # combine b with SV
    elif absorb == 'left':
        l_s, u_s, r_s, d_s = b.shape
        b = b.reshape((l_s, u_s*r_s*d_s))
        U, S, b = svd(b)
        b = b.reshape((-1, u_s, r_s, d_s))
        a = np.einsum('rx,lurd->luxd', np.matmul(U, S), a) # combine a with US 
    else:
        raise ValueError(f"absorb must be either left or right")
    return a, b


def right_canonicalize_mpo(mpo_tensors, start, end):
    # Perform a canonicalization sweep of MPS from left to right.
    mpo_tensors = [tensor.copy() for tensor in mpo_tensors]
    assert end >= start
    for i in range(start, end):
        mpo_tensors[i:i+2] = canonicalize_mpo_tensors(*mpo_tensors[i:i+2], absorb='right')
    return mpo_tensors


def left_canonicalize_mpo(mpo_tensors, start, end):
    # Perform a canonicalization sweep of MPS from right to left.
    mpo_tensors = [tensor.copy() for tensor in mpo_tensors]
    assert start >= end
    for i in range(start, end, -1):
        mpo_tensors[i-1:i+1] = canonicalize_mpo_tensors(*mpo_tensors[i-1:i+1], absorb='left')
    return mpo_tensors


def canonical_center_mpo(mpo, center):
    # right and left canonicalize mps from scratch
    mpo_r = right_canonicalize_mps(mpo, 0, center)
    mpo_rl = left_canonicalize_mps(mpo_r, len(mpo)-1, center)

    return mpo_rl

def shift_canonical_center_mpo(mpo, center, initial=None):
    # shifts canonical center from initial site
    if initial == None:
        return canonical_center_mpo(mpo, center)
    elif initial > center:
        mpo = [tensor.copy() for tensor in mpo]
        for i in range(initial, center, -1):
            mpo[i-1:i+1] = canonicalize_mpo_tensors(*mpo[i-1:i+1], absorb='left')
        return mpo
    else:
        mpo = [tensor.copy() for tensor in mpo]
        for i in range(initial, center):
            mpo[i:i+2] = canonicalize_mpo_tensors(*mpo[i:i+2], absorb='right')
        return mpo

def compress_mps(input_mps, chi, curr_center=None):
    mps = [tensor.copy() for tensor in input_mps]
    mps = shift_canonical_center(mps, 0, curr_center)
    n = len(mps)
    t = mps[0]
    for i in range(1, n):
        l, p, r = t.shape
        t = t.reshape((l*p, r))
        U, S, V = svd(t, chi)
        mps[i-1] = U.reshape((l, p, -1))
        t = np.einsum('ab, bpr->apr', np.matmul(S, V).reshape((-1, r)), mps[i])
    mps[-1] = t
    for i in range(n-2, -1, -1):
        l, p, r = t.shape
        t = t.reshape((l, p*r))
        U, S, V = svd(t, chi)
        mps[i+1] = (V.reshape((-1, p, r)))
        t = np.einsum('ab, lpa->lpb', np.matmul(U, S).reshape((l, -1)), mps[i])
    mps[0] = t
    
    return mps

def compress_mpo(input_mpo, chi, curr_center=None):
    mpo = [tensor.copy() for tensor in input_mpo]
    mpo = shift_canonical_center_mpo(mpo, 0, curr_center)
    n = len(mpo)
    t = mpo[0]
    for i in range(1, n):
        l, p, r = t.shape
        t = t.reshape((l*p, r))
        U, S, V = svd(t, chi)
        mpo[i-1] = U.reshape((l, p, -1))
        t = np.einsum('ab, bpr->apr', np.matmul(S, V).reshape((-1, r)), mpo[i])
    mpo[-1] = t
    for i in range(n-2, -1, -1):
        l, p, r = t.shape
        t = t.reshape((l, p*r))
        U, S, V = svd(t, chi)
        mpo[i+1] = (V.reshape((-1, p, r)))
        t = np.einsum('ab, lpa->lpb', np.matmul(U, S).reshape((l, -1)), mpo[i])
    mpo[0] = t
    
    return mpo

def multiply_mps_mpo(input_mps, input_mpo, chi=None):
    # peforms mps mpo multiplication
    mps = [tensor.copy() for tensor in input_mps]
    mpo = [tensor.copy() for tensor in input_mpo]
    n = len(mps)
    p = mps[0].shape[1]
    t = np.einsum('ipj,kplm->ijlm', mps[0], mpo[0])
    for i in range(1, n):
        t_mps_mpo = np.einsum('ijlm,jqr,lqsn->imrsn', t, mps[i], mpo[i])
        i_s, m_s, r_s, s_s, n_s = t_mps_mpo.shape
        t_mps_mpo = t_mps_mpo.reshape((i_s*m_s, -1))
        mps_node, S, V = svd(t_mps_mpo, chi)
        t = np.matmul(S, V).reshape((-1, r_s, s_s, n_s))
        mps[i-1] = mps_node.reshape((i_s, m_s, -1))
    t = t.reshape(-1,p,1)
    mps[-1] = t
    
    return mps

def multiply_mpo_mpo(mpo_2, mpo_1, chi=None):
    # peforms mpo mpo multiplication
    n = len(mpo_1)
    p = mpo_1[0].shape[1]
    t = np.einsum('akbp,lprP->akbrP', mpo_1[0], mpo_2[0])
    output_mpo = []
    for i in range(1, n):
        t_mpo_mpo = np.einsum('akbrP,bKcD,rDeF->akPKceF', t, mpo_1[i], mpo_2[i])
        a_s, k_s, P_s, K_s, c_s, e_s, F_s = t_mpo_mpo.shape
        t_mpo_mpo = t_mpo_mpo.reshape((a_s*k_s*P_s, -1))
        mpo_node, S, V = svd(t_mpo_mpo, chi)
        t = np.matmul(S, V).reshape((-1, K_s, c_s, e_s, F_s))
        output_mpo.append(mpo_node.reshape((a_s, k_s, P_s, -1)).transpose((0, 1, 3, 2)))
    t = t.reshape(-1,p,1,p)
    output_mpo.append(t)
    
    return output_mpo

def get_A_index(binary):
    # get index in original array A
    # binary = sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
    # A_index = sig_1^x sig_2^x ... sig_n_bits^x sig_1^y sig_2^y ... sig_n_bits^y
    return int(binary[::2]+binary[1::2], 2)

def convert_to_SF2D(MPS):   
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
    if type(B_vec[0, 0]) == np.complex128:
        A_vec = np.zeros(4**n_bits, dtype=complex).reshape((1, -1))    
    else:
        A_vec = np.zeros(4**n_bits).reshape((1, -1))    

    for _ in range(4**n_bits):
        A_index = get_A_index(w)             
        B_index = int(w, 2)                   
        w = bin(B_index+1)[2:].zfill(2*n_bits)     
        A_vec[0, A_index]  = B_vec[0, B_index]

    return A_vec.reshape((N, N))

def convert_to_MPS2D(A, chi=None):  
    # converts scalar field to scale-resolved MPS matrices
    Nx, Ny = A.shape            # Get number of points (Nx equals Ny)
    n = int(np.log2(Nx))        # Get number of (qu)bits
    A_vec = A.reshape((1, -1))  # Flatten function
    
    # Reshape into scale resolving representation B
    w = '0'*2*n                                 # index sig_1^x sig_1^y ... sig_n_bits^x sig_n_bits^y
    if type(A_vec[0, 0]) == np.complex128:
        B_vec = np.zeros(4**n, dtype=complex).reshape((1, -1))    
    else:
        B_vec = np.zeros(4**n).reshape((1, -1))   

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

def convert_to_MPS1D(A, chi=None):  
    N = A.shape
    n = int(np.log2(N))
    A = A.reshape((1, -1))  # Flatten function

    MPS = []
    node = A
    
    for _ in range(n-1):
        m, n = node.shape
        node = node.reshape((2*m, int(n/2)))
        U, S, V = svd(node, chi)        # svd
        MPS.append(U)                   # save U as first node of MPS
        node = np.matmul(S, V)          # create remaining matrix S*V for next expansion step

    m, n = node.shape
    node = node.reshape((2*m, int(n/2)))
    MPS.append(node)    # add last node to MPS

    return MPS

def convert_to_SF1D(MPS):   
    # converts scale-resolved MPS matrices to scalar field
    n_bits = len(MPS)
    N = 2**n_bits
    node_L = MPS[0]
    for i in range(1, n_bits):
        m, n = node_L.shape
        node_R = MPS[i].reshape((n, -1))
        node_L = np.matmul(node_L, node_R)
        m, n = node_L.shape
        node_L = node_L.reshape((2*m, int(n/2)))
    A_vec = node_L.reshape((1, -1)) 

    return A_vec.reshape(N)

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

def numpy_MPS_2D(A, chi=None):
    mps = convert_to_MPS2D(A, chi)
    mps_numpy = convert_MPS_to_numpy(mps, 4)

    return mps_numpy

def numpy_MPS_1D(A, chi=None):
    mps = convert_to_MPS1D(A, chi)
    mps_numpy = convert_MPS_to_numpy(mps, 2)

    return mps_numpy

def numpy_SF_2D(mps_numpy):
    mps = convert_numpy_to_MPS(mps_numpy)
    A = convert_to_SF2D(mps)

    return A

def numpy_SF_1D(mps_numpy):
    mps = convert_numpy_to_MPS(mps_numpy)
    A = convert_to_SF1D(mps)

    return A

def expand(mpo):
    m = mpo[0]
    for i in range(1, len(mpo)):
        m = np.einsum('lurd, rURD->luURdD', m, mpo[i])
        l, u, U, R, d, D = m.shape
        m = m.reshape((l, u*U, R, d*D))
    l, u, r, d = m.shape

    return m.reshape((u, d))

def is_unitary(mpo):
    m = expand(mpo)
    return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))

def flip_mps(mps):
    mps_flipped = []
    for tensor in mps:
        mps_flipped.append(np.einsum('lpr->rpl', tensor))
    mps_flipped.reverse()

    return mps_flipped

def inverse_mpo(mpo):
    output_mpo = []
    for tensor in mpo:
        output_mpo.append(np.conjugate(np.einsum('lurd->ldru', tensor)))
    
    return output_mpo

def add_mps(mps_1, mps_2):
    result_mps = []
    
    for i, (tensor_1, tensor_2) in enumerate(zip(mps_1, mps_2)):
        l_1, p_1, r_1 = tensor_1.shape
        l_2, p_2, r_2 = tensor_2.shape
        if i == 0:
            l = l_1
            p = p_1
            r = r_1 + r_2

            new_tensor = np.zeros((l, p, r), dtype=tensor_1.dtype)
            new_tensor[:l_1, :, :r_1] = tensor_1
            new_tensor[:l_1, :, r_1:] = tensor_2
        elif i == len(mps_1)-1:
            l = l_1 + l_2
            p = p_1
            r = r_1

            new_tensor = np.zeros((l, p, r), dtype=tensor_1.dtype)
            new_tensor[:l_1, :, :r_1] = tensor_1
            new_tensor[l_1:, :, :r_1] = tensor_2
        else:
            l = l_1 + l_2
            p = p_1
            r = r_1 + r_2
            
            new_tensor = np.zeros((l, p, r), dtype=tensor_1.dtype)
            new_tensor[:l_1, :, :r_1] = tensor_1
            new_tensor[l_1:, :, r_1:] = tensor_2
            
        result_mps.append(new_tensor)
    
    return result_mps

def add_mps_list(mps_list, coeff_list, chi):
    curr_mps = mps_list[0]
    curr_mps[0] *= coeff_list[0]
    for i in range(1, len(mps_list)):
        new_mps = mps_list[i]
        new_mps[0] *= coeff_list[i]
        curr_mps = add_mps(curr_mps, new_mps)
        curr_mps = compress_mps(curr_mps, chi)
    
    return curr_mps

def transpose_mps_2D(mps):
    for i, tensor in enumerate(mps):
        l, p, r = tensor.shape
        tensor = tensor.reshape((l, 2, 2, r))
        mps[i] = np.einsum("lxyr->lyxr", tensor).reshape((l, p, r))
    
    return mps

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
    mpo = [tensor.copy() for tensor in mps]
    for i, tensor in enumerate(mpo):
        mpo[i] = np.einsum('ijk, jlm->ilkm', tensor, k_delta)
    
    return mpo   # return the MPO

def multiply_mps_mps(mps_1, mps_2, chi=None):
    mpo_2 = hadamard_product_MPO(mps_2)
    mps = multiply_mps_mpo(mps_1, mpo_2, chi)

    return mps