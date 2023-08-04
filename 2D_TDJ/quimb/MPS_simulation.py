from CFD_2D_TDJ import *


# Set parameters
n_bits = 4                  # number of bits per spatial dimension
N = 2**n_bits               # number of gridpoints per spatial dimension
L = 1                       # length of simulation area
chi = 16                    # maximum bond dimension for mps
chi_mpo = chi               # maximum bond dimension for mpos
dt = 0.1*2**-(n_bits-1)     # time step
T = 2                       # final time
dx = 1 / (2**n_bits - 1)    # spacing between grid points
mu = 2.5 * 10**5            # penalty factor
Re = 0.001*200*1e3          # Reynolds number
# Re = 200*1e3


def main():
    U_MPS, V_MPS, U_arrays, V_arrays = build_initial_fields(n_bits, L, chi)
    time_evolution(U_MPS, V_MPS, chi, chi_mpo, dt, T, Re, mu, '/Users/q556220/dev/TN_CFD/2D_TDJ/quimb/data')

if __name__ == "__main__":
    main()