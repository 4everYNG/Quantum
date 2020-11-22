import qubovert as qv
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dimod
import time
import dwave_networkx as dnx

from dwave.system import DWaveSampler, AutoEmbeddingComposite
from minorminer import find_embedding
from qubovert import boolean_var
from qubovert.sim import anneal_pubo
from neal import SimulatedAnnealingSampler


def solve_qubo(Q,
               sampler="CPU",  # CPU or QPU
               k=10,
               chain_strength=None):
    """
    Given an upper triangular matrix Q of size NxN, solves the quadratic unconstrained binary
    optimization (QUBO) problem given by

        minimize sum(x[i] * Q[i,j] * x[j]
                     for i in range(N),
                     for j in range(i+1, N))

    Uses dimod.SimulatedAnnealingSampler, which solves the problem k times through simulated
    annealing (on a regular CPU). This method returns the best solution found.
    """
    # assert isinstance(Q, np.ndarray)
    # assert sampler in ["CPU", "QPU"]
    n = Q.shape[0]
    nz = len(Q[Q != 0])
    print("Solving QUBO problem (%d vars, %d nz) on %s..." % (n, nz, sampler))

    start = time.time()
    if sampler == "CPU":
        sampler = dimod.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(Q, num_reads=k)
    else:
        if chain_strength is None:
            chain_strength = int(10 * np.max(np.abs(Q)))
        sampler = AutoEmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))
        response = sampler.sample_qubo(Q, num_reads=k, chain_strength=chain_strength)
    elapsed = time.time() - start

    print("Solved in %.2f seconds" % elapsed)
    solution = min(response.data(["sample", "energy"]), key=lambda s: s.energy)
    return solution, response



## Plot
def show_graph(adjacency_matrix, pos):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    for node in range(len(pos)):
        gr.add_node(node)
    gr.add_edges_from(edges)
    nx.draw(gr, [(x,y) for x,y in pos], node_size=500, with_labels=True)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    lib = spio.loadmat('lib.mat')

    C = lib['M_C']
    A = lib['M_A']
    B = lib['M_B']
    CTYPE = lib['M_CTYPE']
    VARTYPE = lib['M_VARTYPE']

    N = len(VARTYPE)  # All variables from MATLAB are 'C', we can just assume they are 'B'

    x = {i: boolean_var('x%d' % i) for i in range(N)}

    # Objective function, default is minimization
    model = 0
    # min C' * X
    for i in range(N):
        model += C[i, 0] * x[i]

    # Construct constraints. A * X [ "=" | "<=" | ">=" ] B    S:'=' ,  U:'<=',  L: '>='
    N_con = len(CTYPE)
    for i in range(N_con):
        AX = 0
        for j in range(N):
            AX += A[i, j] * x[j]

        if CTYPE[i] == 'S':
            model.add_constraint_eq_zero(AX - B[i, 0], lam=1)
        elif CTYPE[i] == 'U':
            model.add_constraint_lt_zero(AX - B[i, 0] - 1, lam=1)  # f(x) <= B  ~~  f(x) < (B+1)
        elif CTYPE[i] == 'L':
            model.add_constraint_gt_zero(AX - B[i, 0] + 1, lam=1)

    ##########################################
    # Get the QUBO form of the model
    qubo = model.to_qubo()

    # D-Wave accept QUBOs in a different format than qubovert's format
    # to get the qubo in this form, use the .Q property
    dwave_qubo = qubo.Q
    dwave_Q = np.zeros((model.num_binary_variables, model.num_binary_variables))
    for i in dwave_qubo.keys():
        dwave_Q[i[0], i[1]] = dwave_qubo[i]
    print(dwave_Q.shape)


    ####################
    triangle = []
    for i in range(len(dwave_Q)):
        for j in range(len(dwave_Q)):
            if dwave_Q[i,j] > 0:
                triangle.append((i,j))
    print('Size of triangle: ', len(triangle))

    # https://docs.ocean.dwavesys.com/projects/dwave-networkx/en/latest/intro.html
    # D-Wave 2X
    # C = dnx.chimera_graph(12, 12, 4)
    #
    # D-Wave 2000Q
    # C = dnx.chimera_graph(16, 16, 4)

    G = dnx.chimera_graph(100,100,4)
    square = G.edges
    dnx.draw_chimera(G)
    plt.show()

    embedding = find_embedding(triangle, square, random_seed=10)
    print("Embedding Length: ", len(embedding))
    print(embedding)
    a=1




    #####################
    # dwave_solution, response = solve_qubo(dwave_Q, sampler='CPU')
    # # dwave_solution, response = solve_qubo(dwave_Q)
    # qubo_solution = dwave_solution.sample

    # solve with D-Wave code from QUBOVert
    # res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo, num_reads=1000)
    # qubo_solution = res.first.sample
    #
    #
    #
    # ######################
    # # convert the qubo solution back to the solution to the model
    # model_solution = model.convert_solution(qubo_solution)
    #
    # print("Variable assignment:", model_solution)
    # print("Model value:", model.value(model_solution))
    # print("MATLAB Model value:", lib['M_obj'])
    # print("Constraints satisfied?", model.is_solution_valid(model_solution))
    #
    # print("Problem size:")
    # print("%5d x variables" % len(x))
    # print("%5d aux variables" % (len(model_solution) - len(x)))
    # print("%5d variables in total" % (len(model_solution)))
    #
    # ################### Solution processing
    # sorted_solution = dict()
    # P_SOL = []
    # for i in range(N):
    #     sorted_solution['x%d' % i] = model_solution['x%d' % i]
    #     P_SOL.append(model_solution['x%d' % i])
    # P_SOL = np.array(P_SOL)
    #
    # ## MATLAB solution
    # M_RT = lib['M_rt']
    # M_LK = lib['M_x_link']
    # M_BK = lib['M_x_block']
    # M_DG = lib['M_x_dg']
    # M_LD = lib['M_x_load']
    # M_BR = lib['M_x_br']
    # M_ND = lib['M_x_node']
    #
    # ## Python solution
    # P_SOL_toy = P_SOL
    # P_RT = P_SOL_toy[0: M_RT.size].reshape(M_RT.shape).transpose()
    # P_SOL_toy = np.delete(P_SOL_toy, list(range(M_RT.size)))
    # P_DG = P_SOL_toy[0: M_DG.size]
    # P_SOL_toy = np.delete(P_SOL_toy, list(range(M_DG.size)))
    # P_LK = P_SOL_toy[0: M_LK.size]
    # P_SOL_toy = np.delete(P_SOL_toy, list(range(M_LK.size)))
    # P_BK = P_SOL_toy[0: M_BK.size]
    # P_SOL_toy = np.delete(P_SOL_toy, list(range(M_BK.size)))
    #
    # Pos = lib['M_Pos']
    # plt.figure()
    # show_graph(M_RT, Pos)
    # plt.figure()
    # show_graph(P_RT, Pos)
