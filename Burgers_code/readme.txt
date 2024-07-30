#################################
!!!!!!!!!
Important 
06/13/2024
!!!!!!!!!

Code as currently written in parameter_grid.py and parameter_grid_noise_3.py save to the same locations. Results of parameter_grid.py were moved manually to "without_noise" folder before running parameter_grid_noise_3.py.
##################################


+ parameter_grid.py - generates training data and trains quantum models with number of variational quantum layers and number of qubits each varying from 2 to 5.

+ benchmark.py - trains the "benchmark" model from the original Burgers' paper on the generated training data

+ quantum_comparison.py - trains two models based on the best results from parameter_grid.py: "Quantum Removed" which is a copy of the model but without the quantum Keras layer, and "Quantum Replaced" which replaces the quantum Keras layer with a classical layer for each variational quantum layer each with the number of neurons matching the number of qubits.

+ parameter_grid_noise_3.py - trains model based on best parameters from parameter_grid.py but with noise added to qnode: classical noise added as small angle rotation perturbation with precision 0.03 using random number between -1 and 1 (prec * 2 * (np.random.rand() - 0.5)). Originally hardcoded for n_qubits = n_layers = 5. Now edited to retrieve that info automatically, but has not been tested.

+ plot_results.ipynb - creates figures from results: (1) Parameter grid RMSE values in comparison to benchmark model, (2) rainbow plots of solution to Burgers' equation from quantum and classical models, and (3) a rainbow plot that includes the noisy quantum model. Some editing may be required to fix file paths for plotting the parameter grid.


parameter_grid.py must be run before benchmark.py or quantum_comparison.py. As written, results must be moved to "without_noise" folder before running plot_results.ipynb for figures.