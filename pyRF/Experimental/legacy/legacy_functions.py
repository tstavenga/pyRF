
    def _compile_scattering_matrix2(self):
        # first separate all the different space trees from the forest
        # then for every node in a space tree get the total scattering matrix
        # every space tree has a time domain amplitude and a seperate k vector

        # separate the unconnected space graphs

        self.resonator_list = [sp.Resonator(self.space_network.subgraph(c).copy())
                               for c in nx.weakly_connected_components(self.space_network)]

        # print(S)
        # H = G.copy()
        # edges_to_remove = [e for e in H.edges(data=True) if not e['attribute'] == 'attribute1']
        # H.remove_edges_from(edges_to_remove)
        G = self.resonator_list[0].resonator_network
        pos = nx.spring_layout(G)
        # nx.draw_spring(self.space_network, pos)
        nx.draw_networkx(G, pos, width=3, edge_color="r", alpha=0.1)
        ax = plt.gca()
        fig = plt.gcf()
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        imsize = 0.1  # this is the image size
        for z, n in enumerate(G.nodes()):
            (x, y) = pos[n]
            xx, yy = trans((x, y))  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            ax.plot(x, y, '*', markersize=25)

            # a = plt.axes([xa - imsize / 2.0, ya - imsize / 2.0, imsize, imsize])
            # a.imshow(img[n])
            # a.set_aspect('equal')
            # a.axis('off')
        plt.show()
        # k=0
        # for node_name, node in self.element_dict.items():
        #     node._get_scattering_matrix(k)


    def plot_eigenfunction(self):
        global scattering_matrix_dict
        zvals = np.linspace(self.starting_position,self.starting_position+self.length, 100, dtype=np.complex128)
        fig,ax = plt.subplots(1,1)
        self.get_eigenvalue()
        # resonator_eigenfunction = self.eigenfunction()
        ax.plot(zvals,np.abs(self.eigenfunction(zvals)))
        nodes = self.space_network.to_undirected().nodes()
        for node in nodes:
            # print(node + " position: ",scattering_matrix_dict[node].get_position())
            xp = scattering_matrix_dict[node].get_position()
            xp_c = np.array(xp).astype(np.complex128)
            yp = np.abs(self.eigenfunction(xp_c))
            ax.plot(xp,yp,'r.', markersize=10)
            ax.text(xp,
                    yp,
                    node,
                    horizontalalignment='left',
                    verticalalignment='top')
