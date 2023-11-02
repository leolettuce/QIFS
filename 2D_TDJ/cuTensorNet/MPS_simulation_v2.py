from CFD_2D_TDJ_optimized_v2 import *


def main():
    n_bits, N, L, chi, chi_mpo, dt, T, dx, mu, Re, path, solver = load_config('config.json')
    U_MPS, V_MPS, U_arrays, V_arrays = build_initial_fields(n_bits, L, chi)
    time_evolution(U_MPS, V_MPS, chi_mpo, dt, T, Re, mu, path, options, solver=solver)

if __name__ == "__main__":
    main()