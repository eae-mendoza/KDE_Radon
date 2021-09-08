from Radon_KDE import *
from pyrmle import *
stilde_range=np.arange(-10,10,0.01)
hilbert_array=np.imag(hilbert([phi_tilde(s) for s in stilde_range]))


sample = sim_sample(10000,2)

normed_xs = normalized_test=preprocessing.normalize(sample[:,0:2],norm='l2')
xs = sample[:,0:2]
#Creating Y

x_norm=[np.linalg.norm(i) for i in xs]
y_normalized=sample[:,2]/x_norm

xy_sample=np.c_[normalized_test,y_normalized]

#---------------#
#Constant Calculation BLock#
#---------------#
d=2
h_theta=0.1
h=2/40
C_rad=C_radtf(d)

probabilities=fhat_theta(normalized_test,h_theta,0)

#Discretization Block
#Definining end-points
start_point=-1
end_point=1
step_size=(end_point-start_point)/40
dist_range=dt_mtx(start_point,end_point,step_size)
print('Number of steps:' +str(len(dist_range)) + ' n*h: ' + str(h*len(dist_range)))

#Estimating over the whole discretization range
start=time.time()
f_estimates=np.array([[fbetahat(xy_sample,dist_range[i][j],d,h,C_rad,probabilities,stilde_range,hilbert_array) for i in range(0,len(dist_range))] for j in range(0,len(dist_range))])
end=time.time()
print('Total Run Time:'+ str((end-start)/60) + 'mins' + ' Shape of Array:' + str(np.shape(f_estimates)))


x1=np.arange(start_point,end_point,step_size)
x2=np.arange(start_point,end_point,step_size)
X, Y = np.meshgrid(x1,x2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, f_estimates/sum(step_size**2*np.ravel(f_estimates)),cmap='Reds',linewidth=0,alpha=0.9)
ax.set_xlabel('B0 axis')
ax.set_ylabel('B1 axis')
ax.set_zlabel('f_beta axis')
ax.view_init(elev=30, azim=130)
plt.show()




