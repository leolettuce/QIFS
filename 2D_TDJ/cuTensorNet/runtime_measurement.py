from CFD_2D_TDJ import *
import time, pickle


def main():
    # Define parameters
    # n_bits_list = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    n_bits_list = np.array([11])
    L = 1
    # chi_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    chi_list = list(range(5, 55, 5)) + list(range(55, 1055, 50))
    name = 'timings_v4_H100_11.p'
    dt = 0.1*2.0**-(n_bits_list-1)
    T = 10*dt
    mu = 2.5 * 10**5
    Re = 200*1e3
    path = None
    solver = "cg"
    data = {}
    sim = QI_CFD()

    for chi in chi_list:
        for i, n in enumerate(n_bits_list):
            sim.init_params(n, L, chi, max(chi, 32), T, mu, Re, path, solver)
            sim.build_initial_fields()

            start = time.time()
            sim.time_evolution()
            end = time.time()

            data[(n, chi)] = end - start
            print(f"n = {n}, chi = {chi}, time = {np.round(end-start, 3)} s")
            with open(name, 'wb') as fp:
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()