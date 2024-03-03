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
import time, pickle


def main():
    # Define parameters
    n_bits_list = np.array([10, 11])
    L = 1
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