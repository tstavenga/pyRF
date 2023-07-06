import calculation_module as cm
import matplotlib.pyplot as plt

CPW1_params = {'Z0':50,
             'Phi0':0.96e8}
L = 5e-3 # 5 mm
reso1 = (cm.Resonator()
        .add_short('A',z=-L/2)#,L=1e-)
        .add_junction('B',z=0)
        .add_open('C',z=L/2)#,C=25e-15)
        .add_segment(('A','B(0)'), **CPW1_params)
        .add_segment(('B(1)','C'), **CPW1_params))


# reso1._compile_node_scattering_matrix()
smat = reso1._get_scattering_matrix()
# pmat = reso1._get_propagation_matrix()
print(reso1.get_eigenmode(guess=400))
# print('pmat',block_matrix(pmat(reso1.k_res)))
# print('smat',block_matrix(smat(reso1.k_res)))
eigenfunc = reso1.get_eigenfunction()
zvals = np.linspace(-L/4,L/4,100)
fig,ax = plt.subplots(1,1)
print(np.absolute(eigenfunc(0,zvals)))
ax.plot(zvals-L/4,np.absolute(eigenfunc(0,zvals-L/4)))
ax.plot(zvals+L/4,np.absolute(eigenfunc(1,zvals+L/4)))
# plt.plot()
# nx.draw(reso1.R)
plt.show()