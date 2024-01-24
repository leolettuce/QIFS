from CFD_2D_TDJ import *


def main():
    # Define parameters
    n_bits = 13															# number of (qu)bits resulting in a 2^n x 2^n grid
    L = 1																# domain size in x and y
    chi = 100															# maximal bond dimension for MPSs
    chi_mpo = 100														# maximal bond dimension for MPOs
    T = 2																# Final simulation time
    mu = 2.5 * 10**5														# Penalty term
    Re = 200*1e3														# Reynolds number (factor of 200 corresponds to the thickness (1/200) of the shear layer)
    path = '/raid/home/q556220/dev/TN_CFD/2D_TDJ/cuTensorNet/QI_1000_100_13'	# folder for saving flow field for various time steps
    save_number = 100													# number of savings for t in [0, T]
    solver = "cg"															# linear system solver (conjugate gradient "cg", np.linalg.solve "inv", scipy.sparse.linalg.cg "scipy.cg")

    sim = QI_CFD()														# create object of QI_CFD() class
    sim.init_params(n_bits, L, chi, chi_mpo, T, mu, Re, path, solver, save_number)		# initialize parameters
    sim.build_initial_fields()													# initialize velocity field according to the jet/shear layer problem
    sim.time_evolution()													# run the simulation

if __name__ == "__main__":
    main()