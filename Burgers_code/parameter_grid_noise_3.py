import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import qmc # upgrade scipy for LHS !pip install scipy --upgrade
from scipy.io import loadmat
import time
from silence_tensorflow import silence_tensorflow
silence_tensorflow() # to remove typcast warnings ....
tf.keras.backend.set_floatx("float64")
import pennylane as qml
import pickle
from copy import deepcopy
import itertools
import threading
import time
import sys

model_folder = "../../models/"
data_folder = "../../data/"
fig_folder = "../../reports/figures/"




# ### generating data

# # number of boundary and initial data points
# # value `Nd` in the reference paper:
# # Nd = number_of_ic_points + number_of_bc1_points + number_of_bc1_points 
# number_of_ic_points = 50
# number_of_bc1_points = 25
# number_of_bc2_points = 25

# # Latin Hypercube Sampling (LHS) engine ; to sample random points in domain,
# # boundary and initial boundary
# engine = qmc.LatinHypercube(d=1)

# # temporal data points
# t_d = engine.random(n=number_of_bc1_points + number_of_bc2_points)
# temp = np.zeros([number_of_ic_points, 1]) # for IC ; t = 0
# t_d = np.append(temp, t_d, axis=0)
# # spatial data points
# x_d = engine.random(n=number_of_ic_points)
# x_d = 2 * (x_d - 0.5)
# temp1 = -1 * np.ones([number_of_bc1_points, 1]) # for BC1 ; x = -1
# temp2 = +1 * np.ones([number_of_bc2_points, 1]) # for BC2 ; x = +1
# x_d = np.append(x_d, temp1, axis=0)
# x_d = np.append(x_d, temp2, axis=0)

# # output values for data points (boundary and initial)
# y_d = np.zeros(x_d.shape)

# # for initial condition: IC = -sin(pi*x)
# y_d[ : number_of_ic_points] = -np.sin(np.pi * x_d[:number_of_ic_points])

# # all boundary conditions are set to zero
# y_d[number_of_ic_points : number_of_bc1_points + number_of_ic_points] = 0
# y_d[number_of_bc1_points + number_of_ic_points : number_of_bc1_points + number_of_ic_points + number_of_bc2_points] = 0

# # number of collocation points
# Nc = 10000

# # LHS for collocation points
# engine = qmc.LatinHypercube(d=2)
# data = engine.random(n=Nc)
# # set x values between -1. and +1.
# data[:, 1] = 2*(data[:, 1]-0.5)

# # change names
# t_c = np.expand_dims(data[:, 0], axis=1)
# x_c = np.expand_dims(data[:, 1], axis=1)

# # convert all data and collocation points to tf.Tensor
# x_d, t_d, y_d, x_c, t_c = map(tf.convert_to_tensor, [x_d, t_d, y_d, x_c, t_c])

# # Number of epochs for training
# epochs = 2000

### Retrieve data

with open(model_folder+'without_noise/parameter_grid/pg_results.pkl', 'rb') as handle:
    pg_dict = pickle.load(handle)
    
data = pg_dict["train_data"]
x_d = data["x_d"]
t_d = data["t_d"]
y_d = data["y_d"]
x_c = data["x_c"]
t_c = data["t_c"]
epochs = data["epochs"]

n_qubits = pg_dict["best"]["n_qubits"]
n_layers = pg_dict["best"]["n_layers"]

### Data points used for plotting
data = loadmat(data_folder+'burgers_shock.mat')
T = data['t'].flatten()[:,None].flatten()
X = data['x'].flatten()[:,None].flatten()
exact_sol = np.real(data['usol']).T

# Dimensions of data
# n = 100
# m = 256
n,m = exact_sol.shape
X0, T0 = np.meshgrid(X, T)
X = X0.reshape([n*m, 1])
T = T0.reshape([n*m, 1])
X = tf.convert_to_tensor(X)
T = tf.convert_to_tensor(T)


### Convenience Functions

def prog_bar(num, denom, length=20):
    # Convenient function for making a progress bar string
    
    if num <= denom:
        prog = round(num/denom*length)
        return ''.join(["[","="*prog,"-"*(length-prog),"]"])
    else:
        return "No progress bar. {} > {}".format(num,denom)
    
def epoch_prog_bar(epoch, epochs, loss):
    pc = str(round(float(epoch+1)/epochs*100, 2))
    
    if loss is None:
        loss_str = ""
    else:
        loss_str = f"Loss: {round(loss.numpy(),4):<5}"
    
    pbar = ''.join(["\rProgress: ", \
      prog_bar(epoch+1,epochs,length=50), \
      " "+" "*(3-len(pc.split('.')[0]))+pc+"0"*(2-len(pc.split('.')[1]))+"% ", \
      f"{epoch+1:>4}/{epochs}     ", \
      loss_str, \
      "     "])
    
    print(pbar,end='')
    return

def animate():
    # start = time.time()
    for c in itertools.cycle([max(0,i-4)*'-' + min(4,i,8+4-i)*'=' + max(8-i,0)*'-' + ' ' for i in range(8+4+1)]):#['   ','.  ','.. ','...']):
        if done:
            break
        sys.stdout.write('\rWorking ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    # end = time.time()
    sys.stdout.write('\rDone!                              \n\n')
    # sys.stdout.write(f'\nTime to Complete: {end - start:.3} (s)\n')

def argmedian(x):
  return np.argpartition(x, len(x) // 2)[len(x) // 2]




### Set up dictionary for recording results

pg_dict = {}

train_data = {
    "x_d": x_d, 
    "t_d": t_d, 
    "y_d": y_d, 
    "x_c": x_c, 
    "t_c": t_c,
    "epochs": epochs,
}
pg_dict["train_data"] = train_data


plot_data = {
    "X0": X0,
    "T0": T0,
    "exact_sol": exact_sol,
    # "benchmark_rmse": qc_dict['quantum_replaced']['RMSE']
}
pg_dict["plot_data"] = plot_data

pg_dict["description"] = "Results are organized as dict[n_qubits][n_layers][run]"



### Perform grid
p = 0.03 # noise precision
pg_dict["train_data"]["prec"] = p


pg_dict[n_qubits] = {}
        
pg_dict[n_qubits][n_layers] = {}
rmse_list = []

for run in range(5):
    print("#"*30)
    print(f"{n_qubits} qubits, {n_layers} layers, run {run}")
    print("#"*30)
    try:
        done = False
        t = threading.Thread(target=animate)
        t.start() # start animation


        ### Set up model

        tf.keras.backend.clear_session()

        # dev = qml.device("default.mixed", wires=n_qubits)
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface = 'tf')
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            # qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # for wire in range(n_qubits):
            #     qml.DepolarizingChannel(p, wires=wire)
            for i in range(n_qubits):
                qml.Rot(p*2*(np.random.rand() - 0.5), p*2*(np.random.rand() - 0.5), p*2*(np.random.rand() - 0.5), wires=i)
            return [qml.expval(qml.PauliZ(wires=wire)) for wire in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)} # for BasicEntanglerLayers
        # weight_shapes = {"weights": (n_layers, n_qubits, 3)} # for StronglyEntanglingLayers

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(n_qubits,activation="tanh"),
            qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits), 
            tf.keras.layers.Dense(1, activation=None)
        ])

        # u(t, x) just makes working with model easier and the whole code looks more
        # like its mathematical backend
        @tf.function
        def u(t, x):
            # model input shape is (2,) and `u` recieves 2 arguments with shape (1,)
            # to be able to feed those 2 args (t, x) to the model, a shape (2,) matrix
            # is build by simply concatenation of (t, x)
            u = model(tf.concat([t, x], axis=1)) # note the axis ; `column`
            return u

        # the physics informed loss function
        # IMPORTANT: this loss function is used for collocation points
        @tf.function
        def f(t, x):
            u0 = u(t, x)
            u_t = tf.gradients(u0, t)[0]
            u_x = tf.gradients(u0, x)[0]
            u_xx = tf.gradients(u_x, x)[0]
            F = u_t + u0*u_x - (0.01/np.pi)*u_xx
            return tf.reduce_mean(tf.square(F))

        # MSE loss function
        # IMPORTANT: this loss function is used for data points
        @tf.function
        def mse(y, y_):
            return tf.reduce_mean(tf.square(y-y_))


        model.save_weights(model_folder+f"parameter_grid/weights/pg_{n_qubits}qubits_{n_layers}layers_run{run}_initial.weights.h5")

        ### Run training

        loss_list = []

        # L-BFGS optimizer was used in the reference paper
        opt = tf.keras.optimizers.Adam(learning_rate=5e-4) # produces warning about M1/M2 mac
        # opt = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4)
        start = time.time()
        # print("Starting training...",end='')
        # epoch_prog_bar(-1,epochs,loss=None)

        # training loop
        # IMPORTANT: a while-based training loop is more beneficial
        # updates the model while loss > 0.006
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # model output/prediction
                y_ = u(t_d, x_d)
                # physics-informed loss for collocation points
                L1 = f(t_c, x_c)
                # MSE loss for data points
                L2 = mse(y_d, y_)
                loss = L1 + L2
            # compute gradients
            g = tape.gradient(loss, model.trainable_weights)
            loss_list.append(loss)

            # apply gradients
            opt.apply_gradients(zip(g, model.trainable_weights))
            # Print progess bar
            # epoch_prog_bar(epoch,epochs,loss)

        end = time.time()
        # print()
        # print(f"{end - start:.3} (s)")
        # print()

        ### Save model params
        
        S = u(T,X).numpy().reshape(n,m)
        # S = np.zeros(n*m)
        # for i in range(n*m):
        #     S[i] = u(tf.convert_to_tensor([T[i]]),tf.convert_to_tensor([X[i]])).numpy()[0][0]
        # S.reshape(n,m)
        
        rmse = np.sqrt(np.mean((S-exact_sol)**2))

        results = {
            "total_params": np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]),
            "pred_sol": u(T, X).numpy().reshape(n, m),
            "loss": loss_list,
            "RMSE": rmse,
            "time": end - start,
        }
        pg_dict[n_qubits][n_layers][run] = results
        model.save_weights(model_folder+f"parameter_grid/weights/pg_{n_qubits}qubits_{n_layers}layers_run{run}_final.weights.h5")
        rmse_list.append(rmse)

    finally:
        done=True # finish animation loop
    time.sleep(0.1)

# Record median run (w.r.t. RMSE)
med_run = argmedian(rmse_list)
pg_dict[n_qubits][n_layers]["median"] = deepcopy(pg_dict[n_qubits][n_layers][med_run])

# ### Find best median run
# rmse_arr = np.zeros((4,4))
# for i in range(4):
#     for j in range(4):
#         rmse_arr[i][j] = pg_dict[i+2][j+2]["median"]["RMSE"]

# best_i, best_j = np.unravel_index(np.argmin(rmse_arr, axis=None), rmse_arr.shape)
# best_result = deepcopy(pg_dict[best_i+2][best_j+2]["median"])
# best_result["n_qubits"] = best_i+2
# best_result["n_layers"] = best_j+2
# pg_dict["best"] = best_result


### Save results
with open(model_folder+'parameter_grid/pg_results.pkl', 'wb') as handle:
    pickle.dump(pg_dict, handle)