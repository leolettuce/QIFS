from CFD_2D_TDJ_optimized import *


# Create field
n_bits = 10
N = 2**n_bits
L = 1
chi = 32
chi_mpo = 32

# Set timesteps
dt = 0.1*2**-(n_bits-1)
T = 2

# Set penalty factor for breach of incompressibility condition
dx = 1 / (2**n_bits - 1)
mu = 2.5 * 10**5
# mu = 1e3
# Re = 0.001*200*1e3
Re = 200*1e3

# Path for saving images
path = '/home/q541472/dev/TN_CFD/2D_TDJ/cuTensorNet/temp'

# Linear system solver ("cg", "inv", "scipy.cg")
solver = "cg"

def main():
    U_MPS, V_MPS, U_arrays, V_arrays = build_initial_fields(n_bits, L, chi)
    time_evolution(U_MPS, V_MPS, chi_mpo, dt, T, Re, mu, path, options, solver=solver)

if __name__ == "__main__":
    main()