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


### Retrieve data

with open(model_folder+'parameter_grid/pg_results.pkl', 'rb') as handle:
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

data = loadmat(data_folder+'burgers_shock.mat')
T = data['t'].flatten()[:,None].flatten()
X = data['x'].flatten()[:,None].flatten()
exact_sol = np.real(data['usol']).T
# n = 100
# m = 256
n,m = exact_sol.shape
X0, T0 = np.meshgrid(X, T)
X = X0.reshape([n*m, 1])
T = T0.reshape([n*m, 1])
X = tf.convert_to_tensor(X)
T = tf.convert_to_tensor(T)



### Convenience Functions

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
qc_dict = {}

train_data = {
    "x_d": x_d, 
    "t_d": t_d, 
    "y_d": y_d, 
    "x_c": x_c, 
    "t_c": t_c,
    "epochs": epochs,
}
qc_dict["train_data"] = train_data

plot_data = {
    "X0": X0,
    "T0": T0,
    "exact_sol": exact_sol,
}
qc_dict["plot_data"] = plot_data


#################
# Quantum Removed
#################
print("#"*30)
print('Quantum Removed')
print("#"*30)
qc_dict["quantum_removed"] = {}
rmse_list = []
for run in range(5):
    print(f"Run {run}")

    try:
        done = False
        t = threading.Thread(target=animate)
        t.start() # start animation
        ### Set up model

        tf.keras.backend.clear_session()

        # n_qubits = 4 

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(n_qubits,activation="tanh"),
            # qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits), 
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

        model.save_weights(model_folder+f"quantum_comparison/weights/qc_quantum_removed_run{run}_initial.weights.h5")
        
        ### Run training

        loss_list = []

        # L-BFGS optimizer was used in the reference paper
        opt = tf.keras.optimizers.Adam(learning_rate=5e-4) # produces warning about M1/M2 mac
        # opt = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4)
        start = time.time()

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

        end = time.time()
        
        ### Save model params

        S = u(T,X).numpy().reshape(n,m)
        rmse = np.sqrt(np.mean((S-exact_sol)**2))

        results = {
            "total_params": np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]),
            "pred_sol": S,
            "loss": loss_list,
            "RMSE": rmse,
            "time": end - start,
        }
        qc_dict["quantum_removed"][run] = results
        model.save_weights(model_folder+f"quantum_comparison/weights/qc_quantum_removed_run{run}_final.weights.h5")
        rmse_list.append(rmse)
        
    finally:
        done=True # finish animation loop
    time.sleep(0.1)

# Record median run (w.r.t. RMSE)
med_run = argmedian(rmse_list)
qc_dict["quantum_removed"]["median"] = deepcopy(qc_dict["quantum_removed"][med_run])




##################
# Quantum Replaced
##################
print("#"*30)
print("Quantum Replaced")
print("#"*30)
qc_dict["quantum_replaced"] = {}
rmse_list = []
for run in range(5):
    print(f"Run {run}")
    
    try:
        done = False
        t = threading.Thread(target=animate)
        t.start() # start animation
        ### Set up model

        tf.keras.backend.clear_session()

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(20,activation="tanh"),
            tf.keras.layers.Dense(n_qubits,activation="tanh"),
            # qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits), 
            # tf.keras.layers.Dense(1, activation=None)
        ])
        for i in range(n_layers):
            model.add(tf.keras.layers.Dense(n_qubits, activation="tanh"))
        model.add(tf.keras.layers.Dense(1, activation=None))
        

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

        model.save_weights(model_folder+f"quantum_comparison/weights/qc_quantum_replaced_run{run}_initial.weights.h5")
        
        ### Run training

        loss_list = []

        # L-BFGS optimizer was used in the reference paper
        opt = tf.keras.optimizers.Adam(learning_rate=5e-4) # produces warning about M1/M2 mac
        # opt = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4)
        start = time.time()

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

        end = time.time()
        
        ### Save model params

        S = u(T,X).numpy().reshape(n,m)
        rmse = np.sqrt(np.mean((S-exact_sol)**2))

        results = {
            "total_params": np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]),
            "pred_sol": S,
            "loss": loss_list,
            "RMSE": rmse,
            "time": end - start,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
        }
        qc_dict["quantum_replaced"][run] = results
        model.save_weights(model_folder+f"quantum_comparison/weights/qc_quantum_replaced_run{run}_final.weights.h5")
        rmse_list.append(rmse)
        
    finally:
        done=True # finish animation loop
    time.sleep(0.1)

# # Record summary
# with open('report.txt','w') as fh:
#     # Pass the file handle in as a lambda function to make it callable
#     model.summary(print_fn=lambda x: fh.write(x + '\n'))

# Record median run (w.r.t. RMSE)
med_run = argmedian(rmse_list)
qc_dict["quantum_replaced"]["median"] = deepcopy(qc_dict["quantum_replaced"][med_run])


with open(model_folder+'quantum_comparison/qc_results.pkl', 'wb') as handle:
    pickle.dump(qc_dict, handle)