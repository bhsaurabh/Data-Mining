import numpy as np


class PageRank(object):
    """
    Calculates the Page Rank of a web-graph

    Parameters
    ----------
    connections: dict
        The connections of the web pages
        Expects something like: {'a':['b', 'c', 'd'], 'b':['b']}
        NOTE 1: All nodes have to be present as keys
        NOTE 2: The values (list) has to be in ascending order 
                (ex: ['a', 'b', 'c'] and not ['c', 'a', 'b'])

    beta: int
        Teleport factor
    """
    def __init__(self, connections, beta):
        self.connections = connections
        self.beta = beta

    def calculate_stochastic_adjacency(self):
        """
        Use the connections information to calculate the stochastic 
        adjacency matrix for the web graph

        Returns
        -------
        M: numpy.matrix
            The stochastic adjacency matrix
        """
        # Initialize the matrix
        M = []
        # Populate the matrix
        for node in self.connections.keys():
            # this is to populate each row, now check all columns
            arr = []
            for out_node in self.connections.keys():
                if node in self.connections[out_node]:
                    arr.append(1.0 / len(self.connections[out_node]))
                else:
                    arr.append(0)  # TODO: Add space efficiency
            M.append(arr)
        return np.matrix(M)
    
    def calculate_pageranks(self, epsilon):
        """
        Calculate the page ranks of all pages
        Use the space-optimized equations

        Parameters
        ----------
        epsilon: int
            error factor, the algorithm keeps computing till
            |r_new - r_old|1 <= epsilon

        Returns
        -------
        ranks: dict
            The page ranks of all pages normalised to 1]
        """
        # Initialize the rank matrix
        R = [1.0/len(self.connections)] * len(self.connections)
        R = np.transpose(np.matrix(R))
        R_old = np.transpose(np.matrix([0] * len(self.connections)))
        M = self.calculate_stochastic_adjacency()
        # Power Iteration
        while R.sum() - R_old.sum() > epsilon:
            R_old = R
            R = M * R * self.beta
            S = 1 - R.sum()  # residual pagerank
            # distribute S throughout
            additions = [S / (len(self.connections) * 1.0)] * len(self.connections)
            additions = np.transpose(np.matrix(additions))
            R += S
        # pageranks are now computed
        return R
