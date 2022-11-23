# PINN-EBM

This repository contains the code to run experiments from the 'Physics-informed neural networks with unknown measurement noise' paper.

To perform experiments, run the 'pebm.py' file. Settings can be specified in the 'input.py' file. Unless specified otherwise, the file located in the 'results/test' folder will be used.

To run multiple experiments with different settings, use the 'master.py' file. A new folder will be created for each setting, if it does not exist already. If no input file is located in these folders, the 'input.py' file from the 'results' folder will be used.

To run experiments on the Navier-Stokes data, first download the 'cylinder_nektar_wake.mat' file from the github page https://github.com/maziarraissi/PINNs, and save it in the folder 'data'.
