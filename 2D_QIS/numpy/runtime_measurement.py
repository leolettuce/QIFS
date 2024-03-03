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

from QIS_2D_numpy import *
import time, pickle

# Create field
n_bits = np.array([5, 6, 7, 8, 9, 10, 11, 12])
N = 2**n_bits
L = 1
chi_list = [4, 8, 16, 32, 64]
# chi_mpo = 16

# Set timesteps
dt = 0.1*2.0**-(n_bits-1)
T = 10*dt

# Set penalty factor for breach of incompressibility condition
# dx = 1 / (2**n_bits - 1)
mu = 2.5 * 10**5
# Re = 0.001*200*1e3
Re = 200*1e3

# Path for saving images
path = None

# Linear system solver ("cg", "inv", "scipy.cg")
solver = "cg"
data = {}

def main():
    for chi in chi_list:
        for i, n in enumerate(n_bits):
            if chi < 32:
                chi_mpo = 32
            else: 
                chi_mpo = chi
            U_MPS, V_MPS, U_arrays, V_arrays = build_initial_fields(n, L, chi)
            start = time.time()
            time_evolution(U_MPS, V_MPS, chi_mpo, dt[i], T[i], Re, mu, path, solver=solver)
            end = time.time()
            data[(n, chi)] = end - start
            print(f"n = {n}, chi = {chi}, time = {np.round(end-start, 3)} s")
            with open('timings.p', 'wb') as fp:
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()