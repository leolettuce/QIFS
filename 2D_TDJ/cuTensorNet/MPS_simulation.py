from CFD_2D_TDJ import *


# Create field
n_bits = 4
N = 2**n_bits
L = 1
chi = 16
chi_mpo = chi

# Set timesteps
dt = 0.1*2**-(n_bits-1)
T = 2

# Set penalty factor for breach of incompressibility condition
dx = 1 / (2**n_bits - 1)
mu = 2.5 * 10**5
Re = 0.001*200*1e3
# Re = 200*1e3


def main():
    U_MPS, V_MPS, U_arrays, V_arrays = build_initial_fields(n_bits, L, chi)
    time_evolution(U_MPS, V_MPS, chi, chi_mpo, dt, T, Re, mu, '/Users/q556220/dev/TN_CFD/2D_TDJ/data')

if __name__ == "__main__":
    main()