#createfigures
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


#results Dir
dirname="Results"
filebase = "springmass__"
nensemble = 10
nconfig_perlayer = 4
nqbit_trials = 3


#First plot rms vs n parmaters
param_fname = "total_nparams"

nparams  = np.loadtxt(dirname+"/"+filebase+param_fname+".txt")
nparams = nparams.reshape(-1,nensemble)
print(nparams)
#load rmse
rmse_fname = "total_error"
rmse = np.loadtxt(dirname+"/"+filebase+rmse_fname+".txt")



#take average over each runs
print("RMSE SHAPE,", rmse.shape)
rmse = rmse.reshape(-1,nensemble)
print("RMSE SHAPE,", rmse.shape)
rmse_avg = rmse.mean(axis=1)
rmse_std = np.std(rmse,axis=1)
print(rmse_avg)
idx = 0
idx_classic = 0


#also plot classical
filebase = "classic_springmass__"
nensemble = 10
nlayer=[2,3,4,2,3,4,2,3,4,5,2,3]
nneuron=[4,4,4,6,6,6,5,5,5,5,8,8]



nnconfigs = [3,3,4,2]
nconfigs =len(nlayer)


nparams_classic  = np.loadtxt(dirname+"/"+filebase+param_fname+".txt")
nparams_classic = nparams_classic.reshape(-1,nensemble)
#load rmse
rmse_classic = np.loadtxt(dirname+"/"+filebase+rmse_fname+".txt")
rmse_classic = rmse_classic.reshape(-1,nensemble)
rmse_classic_avg = rmse_classic.mean(axis=1)
rmse_classic_std = np.std(rmse_classic,axis=1)
#rearrange
fmap = [0,1,2,6,7,8,9,3,4,5]
rmse_classic_avg = rmse_classic_avg[fmap]
rmse_classic_std = rmse_classic_std[fmap]
nparams_classic = nparams_classic[fmap]
#rewrite
nnconfigs = [3,4,3]
nlayer = [2,3,4,2,3,4,5,2,3,4]
nneuron=[4,4,4,5,5,5,5,6,6,6]
print(nparams_classic.shape)
print(rmse_classic_avg.shape)


'''
also now include noisy results
'''

filebase_noisy = "noisy_springmass__"
nlayers  = np.loadtxt(dirname+"/"+filebase_noisy+"n_layers_store"+".txt")
print("nlayers",nlayers)
nensemble_noisy =[]
temp =nlayers[0]
ctr=0
for a in nlayers[1:]:
	ctr+=1
	if a!=temp:
		nensemble_noisy.append(ctr)
		ctr=0
		temp=a
#last one
ctr+=1
if(ctr!=1):
	nensemble_noisy.append(ctr)
	


nparams_noisy  = np.loadtxt(dirname+"/"+filebase_noisy+param_fname+".txt")

#load rmse
rmse_noisy = np.loadtxt(dirname+"/"+filebase_noisy+rmse_fname+".txt")



#take average over each runs based on nensemble
rmse_avg_noisy = np.zeros(len(nensemble_noisy))
rmse_std_noisy = np.zeros(len(nensemble_noisy))
idx1=0
for a in range(len(nensemble_noisy)):
	idx2=idx1+nensemble_noisy[a]
	temp = np.array(rmse_noisy[idx1:idx2])
	rmse_avg_noisy[a] = temp.mean()
	rmse_std_noisy[a] = np.std(temp)
	idx1=idx2





fig = plt.figure(figsize =(24, 6))
ymin = np.amin(rmse_avg-rmse_std)
ymax = np.amax(rmse_avg+rmse_std)
for i in range(nqbit_trials):
	nconfig_perlayer_classic=nnconfigs[i]
	ax = plt.subplot(1, 3, i+1)
	
	ax.set_ylim([ymin, ymax])

	#plt.errorbar(x, y, yerr = y_error, fmt ='o')
	ax.errorbar(nparams[idx:idx+nconfig_perlayer,0],rmse_avg[idx:idx+nconfig_perlayer],rmse_std[idx:idx+nconfig_perlayer],capsize=10,marker="o",markersize=9, capthick=3,ls="--",linewidth=2.0,color="black",label="QPINN w/o noise")
	ax.errorbar(nparams[idx:idx+nconfig_perlayer,0],rmse_avg_noisy[idx:idx+nconfig_perlayer],rmse_std_noisy[idx:idx+nconfig_perlayer],capsize=10,marker="^",markersize=9, capthick=3,ls="--",linewidth=2.0,color="lightgray",label="QPINN with noise")
	ax.errorbar(nparams_classic[idx_classic:idx_classic+nconfig_perlayer_classic,0],rmse_classic_avg[idx_classic:idx_classic+nconfig_perlayer_classic],rmse_classic_std[idx_classic:idx_classic+nconfig_perlayer_classic],capsize=5,marker="x",markersize=9, linewidth=2.0,capthick=1,ls="-",color="grey",label="PINN")
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.set_ylabel("RMSE", fontsize=18)
	ax.set_xlabel("nparams", fontsize=18)
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	ax.grid()
	
	ax.set_title(str(i+2)+" qubits",fontsize=20)
	idx+=nconfig_perlayer
	idx_classic+=nconfig_perlayer_classic
ax.legend(fontsize="18")
plt.savefig("RMSEvsnparams_classic_vs_quantum_noise.png")
#plt.show()
plt.close() 


'''
fig = plt.figure(figsize =(24, 6))
ymin = np.amin(rmse_classic_avg-rmse_classic_std)
ymax = np.amax(rmse_classic_avg+rmse_classic_std)
idx = 0
for i in range(nqbit_trials):
	nconfig_perlayer=nnconfigs[i]
	ax = plt.subplot(1, 3, i+1)
	ax.set_ylim([ymin, ymax])
	#plt.errorbar(x, y, yerr = y_error, fmt ='o')
	ax.errorbar(nparams_classic[idx:idx+nconfig_perlayer,0],rmse_classic_avg[idx:idx+nconfig_perlayer],rmse_classic_std[idx:idx+nconfig_perlayer],capsize=5,marker="o", capthick=1,ls="--",color="black")
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.set_ylabel("RMSE", fontsize=18)
	ax.set_xlabel("nparams", fontsize=18)
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	ax.grid()
	
	ax.set_title(str(i+2)+" qbits",fontsize=20)
	idx+=nconfig_perlayer
plt.savefig("RMSEvsnparams_classic.png")
plt.show()
'''


#now create plot for solution for each configuration
filebase = "springmass__"
nensemble = 10
nconfig_perlayer = 4
nqbit_trials = 3


#First plot rms vs n parmaters
param_fname = "total_u"

QPINN_u  = np.loadtxt(dirname+"/"+filebase+param_fname+".txt",delimiter=",")
npts = QPINN_u.shape[1]
print("QPINN_U shape",QPINN_u.shape)
QPINN_u = QPINN_u.reshape((nconfig_perlayer*nqbit_trials,nensemble,npts))
print("QPINN_U shape",QPINN_u.shape)





#take average over each ensemble of runs
#rmse = rmse.reshape(-1,nensemble)
QPINN_u_avg = QPINN_u.mean(axis=1)
QPINN_u_std = np.std(QPINN_u,axis=1)


'''
Add noise
'''
#now create plot for solution for each configuration
#First plot rms vs n parmaters
QPINN_u_noisy  = np.loadtxt(dirname+"/"+filebase_noisy+param_fname+".txt",delimiter=",")
npts_noisy = QPINN_u_noisy.shape[1]
print("QPINN_U noisy shape",QPINN_u_noisy.shape)
nensemble_noisy = np.array(nensemble_noisy,dtype=np.int32)
print(nensemble_noisy.sum())

QPINN_u_noisy_avg = np.zeros((nconfig_perlayer*nqbit_trials,npts_noisy))
QPINN_u_noisy_std = np.zeros((nconfig_perlayer*nqbit_trials,npts_noisy))
idx1=0
for i,n in enumerate(nensemble_noisy):
	idx2=idx1+n
	temp = QPINN_u_noisy[idx1:idx2,:]
	QPINN_u_noisy_avg[i,:] = temp.mean(axis=0)
	QPINN_u_noisy_std[i,:] = temp.std(axis=0)
	idx1=idx2






xL = 0.0
xR = 3.0
x_QPINN = np.linspace(xL,xR,npts)
nlayers = [3,5,7,9]

#also plot reference solution
x_ref = np.linspace(xL,xR,100)
u_ref = u = -6*np.exp(-3*x_ref) + 7*np.exp(-2*x_ref) + np.sin(x_ref) - np.cos(x_ref)
fig = plt.figure(figsize =(12, 12))
ctr=1
for i in range(nconfig_perlayer):
	for j in range(nqbit_trials):
		index =  (ctr-1)//nqbit_trials+ nconfig_perlayer*((ctr-1)%nqbit_trials)
		ax = plt.subplot(nconfig_perlayer,nqbit_trials,ctr)
		ax.plot(x_ref,u_ref,  color='black',linewidth=2.0, label='Exact')
		ax.plot(x_QPINN,QPINN_u_avg[index,:],markersize=7,marker="o",  color='black',linewidth=2.0,linestyle=":" ,label='QPINN w/o noise')
		ax.fill_between(x_QPINN, QPINN_u_avg[index,:]-QPINN_u_std[index,:], QPINN_u_avg[index,:]+QPINN_u_std[index,:], color='darkgrey')
		ax.plot(x_QPINN,QPINN_u_noisy_avg[index,:],markersize=5,marker="x", color='grey',linewidth=1.0,linestyle=":" ,label='QPINN with noise')
		ax.fill_between(x_QPINN, QPINN_u_noisy_avg[index,:]-QPINN_u_noisy_std[index,:], QPINN_u_noisy_avg[index,:]+QPINN_u_noisy_std[index,:], hatch="|", color='lightgrey')

		ax.grid()
		if i ==0:
			ax.set_title(str(j+2)+" qubits",fontsize=20)
		if j ==0:
			ax.set_ylabel(str(nlayers[i])+" layers",fontsize=20)
		ctr+=1


#plt.scatter(x,ua,color = 'blue', s = 100, label='U(x)^h')
#plt.scatter(x,total_u[:][9],color = 'black', s = 100, label='QPIML')
#plt.errorbar(x,u_qml_avg,u_qml_std, fmt="o",color = 'black', label='QPIML')
#fig.suptitle("QPIML Results for " r"$\frac{du}{dx} = sin(x)$", fontsize=20)
#fig.suptitle("QPIML Results for " r"$\frac{d^2u}{dx^2} + \frac{du}{dx} = 1$", fontsize=20)
#plt.xlabel("x", fontsize=18)
#plt.ylabel("u(x)", fontsize=18)
#plt.legend(fontsize="18",loc ="lower left")
ax.legend(fontsize="18",bbox_to_anchor=(-0.1,-0.05))
fig.savefig(filebase_noisy + "u_vals.png", format='png', dpi=300)
plt.close()

#bar plot for configurations and percent errror
#now create plot for solution for each configuration
filebase = "springmass__"
nconfig_perlayer = 4
nqbit_trials = 3


#First plot rms vs n parmaters
param_fname = "total_precent_converge"

QPINN_converge  = np.loadtxt(dirname+"/"+filebase+param_fname+".txt",delimiter=",")

print("converged shape",QPINN_converge.shape)

QPINN_noisy_converge  = np.loadtxt(dirname+"/"+filebase_noisy+param_fname+".txt",delimiter=",")







#take average over each ensemble of runs
nlayers = np.array([3,5,7,9])
width = .8

nlayers_str = list(map(str,nlayers))
#also plot reference solution
x_ref = np.linspace(xL,xR,100)
u_ref = u = -6*np.exp(-3*x_ref) + 7*np.exp(-2*x_ref) + np.sin(x_ref) - np.cos(x_ref)
fig = plt.figure(figsize =(24, 6))
ctr=1
index = 0
for i in range(nqbit_trials):
	ax = plt.subplot(1,nqbit_trials,ctr)
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.set_ylim([0,100])
	ax.bar(nlayers-width/2,QPINN_converge[index:index+nconfig_perlayer],width,color='gray',label="QPINN w/o noise")
	ax.bar(nlayers+width/2,QPINN_noisy_converge[index:index+nconfig_perlayer],width,hatch="/", color='lightgray',label="QPINN with noise")
	ax.grid()
	ax.set_title(str(i+2)+" qubits",fontsize=20)
	ax.set_xticks((3, 5, 7, 9))
	ax.set_xticklabels((3, 5, 7, 9))
	if i==0:
		ax.set_ylabel("percent of runs converged",fontsize=20)
		ax.legend(fontsize="18",loc ="upper left")
	if i ==1:
		ax.set_xlabel("# layers",fontsize=20)
	ctr+=1
	index+=nconfig_perlayer



#plt.scatter(x,ua,color = 'blue', s = 100, label='U(x)^h')
#plt.scatter(x,total_u[:][9],color = 'black', s = 100, label='QPIML')
#plt.errorbar(x,u_qml_avg,u_qml_std, fmt="o",color = 'black', label='QPIML')
#fig.suptitle("QPIML Results for " r"$\frac{du}{dx} = sin(x)$", fontsize=20)
#fig.suptitle("QPIML Results for " r"$\frac{d^2u}{dx^2} + \frac{du}{dx} = 1$", fontsize=20)
#plt.xlabel("x", fontsize=18)
#plt.ylabel("u(x)", fontsize=18)

fig.savefig(filebase_noisy + "convergence.png", format='png', dpi=300)
plt.close()