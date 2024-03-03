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

from QIS_2D import *

# Define parameters
n_bits = 10															    # number of (qu)bits resulting in a 2^n x 2^n grid
L = 1																    # domain size in x and y
chi = 100															    # maximal bond dimension for MPSs
chi_mpo = chi														    # maximal bond dimension for MPOs
T = 2																    # Final simulation time
mu = 2.5 * 10**5														# Penalty term
Re = 1e5														        # Reynolds number 
path = f'2D_QIS/results/run_{n_bits}_{chi}_{Re}'	                            # folder for saving flow field for various time steps
save_number = 100													    # number of savings for t in [0, T]
solver = "cg"															# linear system solver (conjugate gradient "cg", np.linalg.solve "inv", scipy.sparse.linalg.cg "scipy.cg")



def main():
    sim = QI_CFD()														                # create object of QI_CFD() class
    sim.init_params(n_bits, L, chi, chi_mpo, T, mu, Re, path, solver, save_number)		# initialize parameters
    sim.build_initial_fields()													        # initialize velocity field according to the Decaying Jet (DJ) scenario
    
    # # Initialize Decaying Turbulence (DT) scenario
    # u_init = np.load("initial_fields_DT/u_initial_10.npy")
    # v_init = np.load("initial_fields_DT/v_initial_10.npy")
    # sim.set_initial_fields(u_init, v_init)

    sim.time_evolution()													            # run the simulation

if __name__ == "__main__":
    main()