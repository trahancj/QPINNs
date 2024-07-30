#springmass1D_generateresults
import numpy as np
import matplotlib.pyplot as plt
import random
import pennylane as qml
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
import time

silence_tensorflow() # to remove typcast warnings ....
tf.keras.backend.set_floatx("float32")
tf.keras.backend.clear_session()


def flatten(xss):
    return [x for xs in xss for x in xs]



###########################################
###########################################
#Global variables
nruns = 10 #number of ensemble members per configuration
max_iterations = 600 
lr = 0.02 #learning rate
nn = 11 #number of collocation points





############################################
############################################
#Required functions





#'''
#Try a spring mass-damper problem which is an IVP
#https://kyleniemeyer.github.io/ME373-book/content/second-order/numerical-methods.html
#u'' + 5u' + 6u = 10sin(x)
#u(0)=0
#u(2pi)=-1
#u'(0) = 5
#u exact = -6*np.exp(-3*time) + 7*np.exp(-2*time) + np.sin(time) - np.cos(time)
u_0=np.array([[0.0]]).reshape(-1,1)
u_0 = tf.constant(u_0,dtype = tf.float32)
x_0=0.0
x_1=3.0
x = np.linspace(x_0, x_1, nn) 



def fun(x):
    u = -6*np.exp(-3*x) + 7*np.exp(-2*x) + np.sin(x) - np.cos(x)
    return u

def cost_fn(t,net):
	
	t = tf.Variable(t, dtype = tf.float32)
	#print(t)
	t_0 = tf.zeros((1,1))
	t_1 = tf.constant(x_1,dtype = tf.float32)*tf.ones((1,1),dtype=tf.float32)
	with tf.GradientTape(persistent = True) as tape:
		tape.watch(t)
		u = net(t)
		u_t = tape.gradient(u, t)

	u_tt = tape.gradient(u_t,t)
	del tape
	ODE_loss = u_tt + tf.constant(5,dtype = tf.float32)*u_t + tf.constant(6,dtype = tf.float32)*u - tf.constant(10,dtype = tf.float32)*tf.math.sin(t)
	#left and right boundary loss, both vals are 0
	Left_boundary_loss = net(t_0) - tf.constant(u_0,dtype = tf.float32)

	#Try Neumann boundary loss
	#try Neumann boundary
	with tf.GradientTape(persistent = True) as tape:
		tape.watch(t_0)
		u0 = net(t_0)[:,0:1]

	u_t0 = tape.gradient(u0,t_0)
	u_t0=tf.cast(u_t0, tf.float32)
	#print(u_t0)
	del tape

	Neumann_loss =  u_t0 - tf.convert_to_tensor([[5.0]],dtype=tf.float32)

	#Right_boundary_loss = net(t_1) - tf.constant(u_1,dtype = tf.float64)
	square_loss = tf.square(ODE_loss) + tf.square(Left_boundary_loss) + tf.square(Neumann_loss)
	total_loss = tf.reduce_mean(square_loss)
	return total_loss

def mse(sol1,sol2):
    return ((sol1 - sol2)**2).mean()

#some global variables to store all the data for each run
total_cost = []
total_u = []
#total_weights = []
total_error = []
total_time = []
total_nparams = []
percent_converge = []
n_qbits_store = []
n_layers_store = []


#analytic solution at collocation points for comparison purposes
ua = fun(x)

train_t = np.zeros((nn,1))
train_t[:,0] = x
###########################################
###########################################
#Loop through for a bunch of runs
for n_qubits in range(2,5):
    for n_layers in range(3,10,2):
        '''
        #establish a qnode device with n_qbits
        dev = qml.device("default.qubit")
        @qml.qnode(dev, diff_method='best')
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        

        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        weight_shapes = {"weights": weight_shape}

        #define a tensorflow architecture
        NN = tf.keras.models.Sequential([ tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(n_qubits,activation="tanh"),
        qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits), 
        tf.keras.layers.Dense(1, activation=None) ])

        '''
        features = tf.Variable(x, dtype=tf.float64)
        irun=0;
        run_count = 0;

        while irun<nruns:
            start =time.time()
            run_count = run_count + 1

            #tf.keras.backend.set_floatx("float64")
            tf.keras.backend.clear_session()

            #doesn't appear it works inside loop so reinitialize each run
            dev = qml.device("default.qubit")
            @qml.qnode(dev, diff_method='best')
            def qnode(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
            weight_shapes = {"weights": weight_shape}

            NN = tf.keras.models.Sequential([ tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(n_qubits,activation="tanh"),
            qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits), 
            tf.keras.layers.Dense(1, activation=None) ])

            #opt=tf.keras.optimizers.legacy.Adam(learning_rate=lr, epsilon=1e-07, amsgrad='False')
            opt = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon=1e-07, amsgrad='False')
            tf.keras.initializers.RandomUniform(minval=0.0,maxval=2*np.pi)
            #np.random.seed(100)
            #weights = tf.keras.initializers.RandomUniform(minval=0, maxval=2*np.pi, seed=200, size=weight_shape)

            #intitialize variables
            #features = tf.Variable(x, dtype=tf.float32)
            
            cost = []



            #run the model
            for itr in range(max_iterations):
                with tf.GradientTape() as tape:
                    train_loss = cost_fn(train_t,NN)
                    cost.append(train_loss)
                    grad_w = tape.gradient(train_loss, NN.trainable_variables)
                    opt.apply_gradients(zip(grad_w, NN.trainable_variables))
                    print("************************************************************************")
                    print(f"RUN: {run_count} || Convergence Count: {irun} || Step = {itr+1} || Cost function = {cost[-1]:.4f} ")
                    print("************************************************************************")
                
                #a condition to expedite discarded runs
                #hard coded
                if itr >= 200 and train_loss > 2.0:
                    print("Run Rejected !!! \n")
                    break
            end = time.time()
            # only accept this run if the model was converging
            if  (nruns > 1 and cost[-1] < 0.3):
                print("**********************************************************************************************")
                print(f"QUBITS: {n_qubits} || LAYERS: {n_layers} || RUN: {run_count} || Convergence Count: {irun} || Cost function = {cost[-1]:.4f} ")
                print("**********************************************************************************************")


                irun = irun+1
                
                #save data to store and eventually write to file
                # Add cost to average over runs
                total_cost.append(cost)
                # Add weights
                #total_weights.append(weights)
                
                #evaluate and add qml solution
                u_qml = NN.predict(x).ravel()
                total_u.append(u_qml)
                #add error at collocation points
                solution_error = mse(ua,u_qml)
                total_error.append([solution_error])

                #save time for each run and nparams for each run
                total_time.append([end-start])
                total_nparams.append([np.sum([np.prod(v.get_shape().as_list()) for v in NN.trainable_variables ])])
                
                n_qbits_store.append([n_qubits])
                n_layers_store.append([n_layers])

        
        convergence_ratio = (nruns/float(run_count))*100
        print(f"Percentage of runs converged: {convergence_ratio:.3f}")
        #store this number
        percent_converge.append(convergence_ratio)

        solution_scale=1
        #Plot some things as we go to look at what is happening
        filebase = "springmass__n_qubits_" + str(n_qubits) + "__n_layers_" + str(n_layers) + "__solution_scale_" + str(solution_scale) + "__lr_" + str(lr) + "__nruns_" + str(nruns)
        '''
        xx = np.linspace(x_0, x_1, 100)
        ff = fun(xx)
        fig = plt.figure(figsize =(8, 20))
        for i in range(0,nruns):
            ax = plt.subplot(5, 2, i+1)
            #ax.set_title(("Run" + str(i+1))) #, fontsize=20)
            ax.text(.38, .1, "Run" + str(i+1), fontsize = 18,  transform = ax.transAxes) #horizontalalignment="center", verticalalignment="bottom")
            ax.plot(xx, ff,  color='black',linewidth=2.0, label='U(x)');
            ax.scatter(x,total_u[:][i],color = 'black', s = 100, label='QPIML')
            if (i % 2) == 0:
                ax.set_ylabel("u(x)", fontsize=18)
            if (i==8 or i==9):
                ax.set_xlabel("x", fontsize=18)
            #ax.xticks(fontsize=18)
            #ax.yticks(fontsize=18)
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
            ax.grid()
            ax.titlesize = 20   # fontsize of the axes title
            ax.labelsize = 18 # fontsize of the x any y labels
        fig.savefig(filebase + "_all_u.jpg", format='jpg', dpi=300)
        #plt.show()
        plt.close()


        
        #Other plots for future reference
        ########################################################
        ########################################################
        # Put data in form I want for easy mean/std calculation
        u_runs = [[0] * nn for i in range(nruns)]
        cost_runs = [[0] * max_iterations for i in range(nruns)]
        #print(uu)
        for j in range(0,nruns):
            for i in range(0,nn):
                u_runs[j][i] = total_u[j][i].numpy()
            for i in range(0,max_iterations):
                cost_runs[j][i] = total_cost[j][i].numpy()
        u_runs = np.array(u_runs)
        cost_runs = np.array(cost_runs)

        ########################################################
        ########################################################
        ########################################################
        ########################################################
        # Calculate mean and STD
        u_qml_avg = np.zeros(nn)
        u_qml_std = np.zeros(nn)
        cost_avg = np.zeros(max_iterations)
        cost_std = np.zeros(max_iterations)
        for i in range(0,nn):
            u_qml_avg[i] = np.median(u_runs[:,i])
            u_qml_std[i] = np.std(u_runs[:,i])
        for i in range(0,max_iterations):
            cost_avg[i] = np.mean(cost_runs[:,i])
            cost_std[i] = np.std(cost_runs[:,i])

        print("u_qml_avg: ",u_qml_avg)
        print("u_qml_std: ",u_qml_std)

        ########################################################
        ########################################################
        # Plot Mean Solution with Variances
        fig = plt.figure(figsize =(6, 6))
        ax = plt.axes()
        xx = np.linspace(x_0, x_1, 100)
        ff = fun(xx)
        ax.plot(xx, ff,  color='black',linewidth=2.0, label='U(x)');
        #plt.scatter(x,ua,color = 'blue', s = 100, label='U(x)^h')
        #plt.scatter(x,total_u[:][9],color = 'black', s = 100, label='QPIML')
        plt.errorbar(x,u_qml_avg,u_qml_std, fmt="o",color = 'black', label='QPIML')
        #fig.suptitle("QPIML Results for " r"$\frac{du}{dx} = sin(x)$", fontsize=20)
        #fig.suptitle("QPIML Results for " r"$\frac{d^2u}{dx^2} + \frac{du}{dx} = 1$", fontsize=20)
        plt.xlabel("x", fontsize=18)
        plt.ylabel("u(x)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.grid()
        plt.legend(fontsize="18",loc ="lower center")
        fig.savefig(filebase + ".jpg", format='jpg', dpi=300)
        plt.close()
        #plt.show()

        ########################################################
        ########################################################
        # Plot Mean Cost with Variances
        fig = plt.figure(figsize =(6, 6))
        plt.plot(range(len(cost)), cost_avg, label='loss',  color='black',linewidth=2.0)
        plt.fill_between(range(len(cost)), cost_avg-cost_std, cost_avg+cost_std, color='grey')
        plt.grid()
        #fig.suptitle("QPIML Loss for " r"$\frac{du}{dx} = sin(x)$", fontsize=20)
        #fig.suptitle("QPIML Loss for " r"$\frac{d^2u}{dx^2} + \frac{du}{dx} = 1$", fontsize=20)
        plt.xlim([0, max_iterations])
        plt.xlabel('epoch', fontsize=18)
        plt.ylabel('loss', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        #plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        fig.savefig(filebase + "_loss.jpg", format='jpg', dpi=300)
        plt.close()
        #plt.show()

        ########################################################
        ########################################################
        # Plot LOG10 Mean Cost with Variances
        fig = plt.figure(figsize =(6, 6))
        plt.plot(range(len(cost)), 10*np.log10(cost_avg), label='loss',  color='black',linewidth=2.0)
        plt.fill_between(range(len(cost)), 10*np.log10(cost_avg)-10*np.log10(cost_std), 10*np.log10(cost_avg)+10*np.log10(cost_std), color='grey')
        plt.grid()
        #fig.suptitle("QPIML Loss for " r"$\frac{du}{dx} = sin(x)$", fontsize=20)
        #fig.suptitle("QPIML LOG10 Loss for " r"$\frac{d^2u}{dx^2} + \frac{du}{dx} = 1$", fontsize=20)
        plt.xlim([0, max_iterations])
        plt.xlabel('epoch', fontsize=18)
        plt.ylabel('loss', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        #plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        fig.savefig(filebase + "_LOG10_loss.jpg", format='jpg', dpi=300)
        plt.close()
        #plt.show()
        '''

# Save all data
filebase = "springmass__"
#firs the total cost
total_cost = np.array(total_cost)
np.savetxt(filebase+"total_cost.txt",total_cost,fmt='%.5f',delimiter=',')
#then the total u
total_u = np.array(total_u)
np.savetxt(filebase+"total_u.txt",total_u,fmt='%.5f',delimiter=',')
#then the total weights
#total_weights = np.array(total_weights)
#np.savetxt(filebase+"total_weights.txt",total_weights,fmt='%.5f',delimiter=',')
#total error
total_error = np.array(total_error)
np.savetxt(filebase+"total_error.txt",total_error,fmt='%.5f',delimiter=',')
#total time
total_time = np.array(total_time)
np.savetxt(filebase+"total_time.txt",total_time,fmt='%.5f',delimiter=',')
#total nparams
total_nparams = np.array(total_nparams)
np.savetxt(filebase+"total_nparams.txt",total_nparams,fmt='%.5f',delimiter=',')
#total precent vonverge
percent_converge = np.array(percent_converge)
np.savetxt(filebase+"total_precent_converge.txt",percent_converge,fmt='%.5f',delimiter=',')
#total n qbits
n_qbits_store = np.array(n_qbits_store)
np.savetxt(filebase+"n_qbits_store.txt",n_qbits_store,fmt='%.5f',delimiter=',')
#total n layers
n_layers_store = np.array(n_layers_store)
np.savetxt(filebase+"n_layers_store.txt",n_layers_store,fmt='%.5f',delimiter=',')





