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

from QIS_2D_quimb import *


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
dx = 1 / (2**n_bits - 1)
dt = 0.1*2**-(n_bits-1)


def main():
    U_MPS, V_MPS, U_arrays, V_arrays = build_initial_fields(n_bits, L, chi)
    time_evolution(U_MPS, V_MPS, chi, chi_mpo, dt, T, Re, mu, path, solver=solver)

if __name__ == "__main__":
    main()