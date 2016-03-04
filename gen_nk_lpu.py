#!/usr/bin/env python

"""
Create generic LPU and simple pulse input signal.
"""

from itertools import product
import sys

import numpy as np
import h5py
import networkx as nx

def create_lpu_graph(lpu_name, N_sensory, N_local, N_proj):
    """
    Create a generic LPU graph comprising spiking neurons.

    Creates a graph containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The graph
    also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are Leaky Integrate-and-Fire neurons, and all
    synapses use the alpha function model.

    Parameters
    ----------
    lpu_name : str
        Name of LPU. Used in port identifiers.
    N_sensory : int
        Number of sensory neurons.
    N_local : int
        Number of local neurons.
    N_proj : int
        Number of project neurons.

    Returns
    -------
    g : networkx.MultiDiGraph
        Generated graph.
    """

    # Set numbers of neurons:
    neu_type = ('sensory', 'local', 'proj')
    neu_num = (N_sensory, N_local, N_proj)

    # Neuron ids are between 0 and the total number of neurons:
    G = nx.DiGraph()
    G.add_nodes_from(range(sum(neu_num)))

    idx = 0
    spk_out_id = 0
    for (t, n) in zip(neu_type, neu_num):
        for i in range(n):
            name = t+"_"+str(i)

            G.node[idx] = {
                'model': 'LeakyIAF',
                'name': name+'_s',
                'extern': True if t == 'sensory' else False, # True if the neuron can receive external input
                'public': True if t == 'proj' else False,    # True if the neuron can emit output
                'spiking': True,
                'V': np.random.uniform(-0.06,-0.025),
                'Vr': -0.0675489770451,
                'Vt': -0.0251355161007,
                'R': 1.02445570216,
                'C': 0.0669810502993}

            # Projection neurons are all assumed to be attached to output
            # ports (which are not represented as separate nodes):
            if t == 'proj':
                G.node[idx]['selector'] = '/%s/out/spk/%s' % (lpu_name, str(spk_out_id))
                G.node[idx]['circuit'] = 'proj'
                spk_out_id += 1
            else:
                G.node[idx]['circuit'] = 'local'

            idx += 1

    # An input port node is created for and attached to each non-projection
    # neuron with a synapse; this assumes that data propagates from one LPU to
    # another as follows:
    # LPU0[projection neuron] -> LPU0[output port] -> LPU1[input port] -> 
    # LPU1[synapse] -> LPU1[non-projection neuron]
    spk_in_id = 0
    gpot_in_id = 0
    for i, data in G.nodes_iter(True):
        if data['public'] == False:
            G.add_node(idx, {
                'name': 'port_in_spk_%s' % spk_in_id,
                'model': 'port_in_spk',
                'selector': '/%s/in/spk/%s' % (lpu_name, idx),
                'spiking': True,
                'public': False,
                'extern': False,
                'circuit': G.node[i]['circuit']
            })
            spk_in_id += 1
            G.add_edge(idx, i, attr_dict={
                'name': G.node[idx]['name']+'-'+G.node[i]['name'],
                'model': 'AlphaSynapse',
                'class': 0,
                'conductance': True,
                'ad': 0.19*1000,
                'ar': 1.1*100,
                'gmax': 0.003,
                'reverse': 0.065,
                'circuit': G.node[i]['circuit']})

        idx += 1

    # Assume a probability of synapse existence for each group of synapses:
    # sensory -> local, sensory -> projection, local -> projection, 
    # projection -> local:            
    for r, (i, j) in zip((0.5, 0.1, 0.1, 0.3),
                         ((0, 1), (0, 2), (1, 2), (2, 1))):
        src_off = sum(neu_num[0:i])
        tar_off = sum(neu_num[0:j])

        for src, tar in product(range(src_off, src_off+neu_num[i]),
                                range(tar_off, tar_off+neu_num[j])):

            # Don't connect all neurons:
            if np.random.rand() > r: continue

            # Connections from the sensory neurons use the alpha function model;
            # all other connections use the power_gpot_gpot model:
            name = G.node[src]['name'] + '-' + G.node[tar]['name']
            G.add_edge(src, tar, attr_dict={
                'model'       : 'AlphaSynapse',
                'name'        : name,
                'class'       : 0 if G.node[tar]['spiking'] is True else 1,
                'ar'          : 1.1*1e2,
                'ad'          : 1.9*1e3,
                'reverse'     : 65*1e-3 if G.node[tar]['spiking'] else 0.01,
                'gmax'        : 3*1e-3 if G.node[tar]['spiking'] else 3.1e-4,
                'conductance' : True,
                'circuit'     : G.node[src]['circuit']})

    return G

def create_lpu(file_name, lpu_name, N_sensory, N_local, N_proj):
    """
    Create a generic LPU graph comprising spiking neurons.

    Creates a GEXF file containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The graph
o    also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are Leaky Integrate-and-Fire neurons, and all
    synapses use the alpha function model.

    Parameters
    ----------
    file_name : str
        Output GEXF file name.
    lpu_name : str
        Name of LPU. Used in port identifiers.
    N_sensory : int
        Number of sensory neurons.
    N_local : int
        Number of local neurons.
    N_proj : int
        Number of project neurons.

    Returns
    -------
    g : networkx.MultiDiGraph
        Generated graph.
    """

    g = create_lpu_graph(lpu_name, N_sensory, N_local, N_proj)
    nx.write_gexf(g, file_name)

def create_input(file_name, N_sensory, dt=1e-4, dur=1.0, start=0.3, stop=0.6, I_max=0.6):
    """
    Create input stimulus for sensory neurons in artificial LPU.

    Creates an HDF5 file containing input signals for the specified number of
    neurons. The signals consist of a rectangular pulse of specified duration
    and magnitude.

    Parameters
    ----------
    file_name : str
        Name of output HDF5 file.
    N_sensory : int
        Number of sensory neurons.
    dt : float
        Time resolution of generated signal.
    dur : float
        Duration of generated signal.
    start : float
        Start time of signal pulse.
    stop : float
        Stop time of signal pulse.
    I_max : float
        Pulse magnitude.
    """

    Nt = int(dur/dt)
    t  = np.arange(0, dt*Nt, dt)

    I  = np.zeros((Nt, N_sensory), dtype=np.float64)
    I[np.logical_and(t>start, t<stop)] = I_max

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('array', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)

def create_input_const(file_name, N_sensory, dt=1e-4, dur=1.0, I_max=0.6):
    """
    Create constant input stimulus for sensory neurons in artificial LPU.

    Creates an HDF5 file containing input signals for the specified number of
    neurons. The signals consist of a constant value.

    Parameters
    ----------
    file_name : str
        Name of output HDF5 file.
    N_sensory : int
        Number of sensory neurons.
    dt : float
        Time resolution of generated signal.
    dur : float
        Duration of generated signal.
    I_max : float
        Signal magnitude.
    """

    Nt = int(dur/dt)
    t  = np.arange(0, dt*Nt, dt)

    I  = np.zeros((Nt, N_sensory), dtype=np.float64)
    I[:] = I_max

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('array', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='nk_lpu.gexf.gz',
                        help='LPU file name')
    parser.add_argument('-n', type=int, nargs=3, default=(30, 30, 30),
                         help='Numbers of sensory, local, and projection neurons')
    parser.add_argument('-s', type=int,
                        help='Seed random number generator')
    parser.add_argument('-l', '--lpu', type=str, default='gen',
                        help='LPU name')

    args = parser.parse_args()

    if args.s is not None:
        np.random.seed(args.s)
    neu_num = args.n

#    create_input_const(args.in_file_name, neu_num[0], dt, dur, I_max)
    create_lpu(args.lpu_file_name, args.lpu, *neu_num)
