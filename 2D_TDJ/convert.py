import quimb as qu
import cupy as cp

def convert_cupy_tensor_to_quimb(cupy_tensor, inds):
    return qu.tensor.Tensor(cupy_tensor.get(), inds=inds)


def convert_quimb_tensor_to_cupy(quimb_tensor):
    return cp.asarray(quimb_tensor.data)


def convert_cupy_mps_to_quimb(cupy_mps):
    arrays = []
    for i, array in enumerate(cupy_mps):
        if i == 0:
            l, p, r = array.shape
            arrays.append(array.get().reshape(p, r).transpose())
        elif i == len(cupy_mps)-1:
            l, p, r = array.shape
            arrays.append(array.get().reshape(l, p))
        else:
            arrays.append(array.get().transpose((0, 2, 1)))

    return qu.tensor.MatrixProductState(arrays, shape="lrp")


def convert_quimb_mps_to_cupy(quimb_mps):
    arrays = []
    for i, tensor in enumerate(quimb_mps):
        if i == 0:
            r, p = tensor.shape
            arrays.append(cp.asarray(tensor.data.reshape((1, r, p)).transpose((0, 2, 1))))
        elif i == quimb_mps.L-1:
            l, p = tensor.shape
            arrays.append(cp.asarray(tensor.data.reshape((l, 1, p)).transpose((0, 2, 1))))
        else:
            arrays.append(cp.asarray(tensor.data.transpose((0, 2, 1))))
    
    return arrays


def convert_cupy_mpo_to_quimb(cupy_mpo):
    arrays = []
    for i, array in enumerate(cupy_mpo):
        if i == 0:
            l, u, r, d = array.shape
            arrays.append(array.get().reshape(u, r, d).transpose((1, 0, 2)))
        elif i == len(cupy_mpo)-1:
            l, u, r, d = array.shape
            arrays.append(array.get().reshape(l, u, d))
        else:
            arrays.append(array.get().transpose((0, 2, 1, 3)))
    
    return qu.tensor.MatrixProductOperator(arrays, shape="lrud")


def convert_quimb_mpo_to_cupy(quimb_mpo):
    arrays = []
    for i, tensor in enumerate(quimb_mpo):
        if i == 0:
            r, u, d = tensor.shape
            arrays.append(cp.asarray(tensor.data.reshape((1, r, u, d)).transpose((0, 2, 1, 3))))
        elif i == quimb_mpo.L-1:
            l, u, d = tensor.shape
            arrays.append(cp.asarray(tensor.data.reshape((l, 1, u, d)).transpose((0, 2, 1, 3))))
        else:
            arrays.append(cp.asarray(tensor.data.transpose((0, 2, 1, 3))))
    
    return arrays