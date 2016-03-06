#!/usr/bin/env python

import argparse
import itertools
from itertools import product
import time

import networkx as nx
import numpy as np
import pycuda.driver as drv

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU

import neurokernel.mpi_relaunch

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

class MyLPU(LPU):
    def __init__(self, dt, n_dict, s_dict, I_const=0.6,
                 output_file=None, device=0, ctrl_tag=core.CTRL_TAG,
                 gpot_tag=core.GPOT_TAG, spike_tag=core.SPIKE_TAG,
                 rank_to_id=None, routing_table=None, id=None, debug=False,
                 columns=['io', 'type', 'interface'], cuda_verbose=False,
                 time_sync=False):
        super(MyLPU, self).__init__(dt, n_dict, s_dict, None, output_file,
                 device, ctrl_tag, gpot_tag,
                 spike_tag, rank_to_id, routing_table,
                 id, debug, columns,
                 cuda_verbose, time_sync)

        self.I_const = I_const

        # Append outputs to list to avoid disk I/O slowdown:
        self.output_gpot_buffer = []
        self.output_spike_buffer = []

    def _write_output(self):
        """
        Save neuron states or spikes to output file.
        The order is the same as the order of the assigned ids in gexf
        """

        if self.total_num_gpot_neurons > 0:
            # self.output_gpot_buffer.append(
            #     self.V.get()[self.gpot_order_l].reshape((-1,)))
            self.output_gpot_buffer.append(
                self.V_host[self.gpot_order_l].reshape((-1,)))
        if self.total_num_spike_neurons > 0:
            # self.output_spike_buffer.append(
            #     self.spike_state.get()[self.spike_order_l].reshape((-1,)))
            self.output_spike_buffer.append(
                self.spike_state_host[self.spike_order_l].reshape((-1,)))

    def post_run(self):        
        super(LPU, self).post_run()
        if self.output:
            if self.total_num_gpot_neurons > 0:
                self.output_gpot_file.root.array.append(np.asarray(self.output_gpot_buffer))
                self.output_gpot_file.close()
            if self.total_num_spike_neurons > 0:
                self.output_spike_file.root.array.append(np.asarray(self.output_spike_buffer))
                self.output_spike_file.close()
        if self.debug:
            # for file in self.in_gpot_files.itervalues():
            #     file.close()
            if self.total_num_gpot_neurons > 0:
                self.gpot_buffer_file.close()
            if self.total_synapses + len(self.input_neuron_list) > 0:
                self.synapse_state_file.close()

        for neuron in self.neurons:
            neuron.post_run()
            if self.debug and not neuron.update_I_override:
                neuron._BaseNeuron__post_run()

        for synapse in self.synapses:
            synapse.post_run()
        
    def _set_constant_input(self):
        # Since I_ext is constant, we can just copy it into synapse_state:
        cuda.memcpy_dtod(
            int(int(self.synapse_state.gpudata) +
                self.total_synapses*self.synapse_state.dtype.itemsize),
            int(self.I_ext.gpudata),
            self.num_input*self.synapse_state.dtype.itemsize)

    def run_step(self):
        super(LPU, self).run_step()

        self._read_LPU_input()

        self._set_constant_input()

        if not self.first_step:
            for i,neuron in enumerate(self.neurons):
                neuron.update_I(self.synapse_state.gpudata)
                neuron.eval()

            self._update_buffer()

            for synapse in self.synapses:
                if hasattr(synapse, 'update_I'):
                    synapse.update_I(self.synapse_state.gpudata)
                synapse.update_state(self.buffer)

            self.buffer.step()
        else:
            self.first_step = False

        if self.debug:
            if self.total_num_gpot_neurons > 0:
                self.gpot_buffer_file.root.array.append(
                    self.buffer.gpot_buffer.get()
                        .reshape(1, self.gpot_delay_steps, -1))
            
            if self.total_synapses + len(self.input_neuron_list) > 0:
                self.synapse_state_file.root.array.append(
                    self.synapse_state.get().reshape(1, -1))

        self._extract_output()

        # Save output data to disk:
        if self.output:
            self._write_output()

    def _initialize_gpu_ds(self):
        """
        Setup GPU arrays.
        """

        self.synapse_state = garray.zeros(
            max(int(self.total_synapses) + len(self.input_neuron_list), 1),
            np.float64)

        if self.total_num_gpot_neurons>0:
            # self.V = garray.zeros(
            #     int(self.total_num_gpot_neurons),
            #     np.float64)
            self.V_host = drv.pagelocked_zeros(
                int(self.total_num_gpot_neurons),
                np.float64, mem_flags=drv.host_alloc_flags.DEVICEMAP)
            self.V = garray.GPUArray(self.V_host.shape,
                                     self.V_host.dtype,
                                     gpudata=self.V_host.base.get_device_pointer())
        else:
            self.V = None

        if self.total_num_spike_neurons > 0:
            # self.spike_state = garray.zeros(int(self.total_num_spike_neurons),
            #                                 np.int32)
            self.spike_state_host = drv.pagelocked_zeros(int(self.total_num_spike_neurons),
                            np.int32, mem_flags=drv.host_alloc_flags.DEVICEMAP)
            self.spike_state = garray.GPUArray(self.spike_state_host.shape,
                                               self.spike_state_host.dtype,
                                               gpudata=self.spike_state_host.base.get_device_pointer())
        self.block_extract = (256, 1, 1)
        if len(self.out_ports_ids_gpot) > 0:
            self.out_ports_ids_gpot_g = garray.to_gpu(self.out_ports_ids_gpot)
            self.sel_out_gpot_ids_g = garray.to_gpu(self.sel_out_gpot_ids)

            self._extract_gpot = self._extract_projection_gpot_func()

        if len(self.out_ports_ids_spk) > 0:
            self.out_ports_ids_spk_g = garray.to_gpu(
                (self.out_ports_ids_spk).astype(np.int32))
            self.sel_out_spk_ids_g = garray.to_gpu(self.sel_out_spk_ids)

            self._extract_spike = self._extract_projection_spike_func()

        if self.ports_in_gpot_mem_ind is not None:
            inds = self.sel_in_gpot_ids
            self.inds_gpot = garray.to_gpu(inds)

        if self.ports_in_spk_mem_ind is not None:
            inds = self.sel_in_spk_ids
            self.inds_spike = garray.to_gpu(inds)

    def _init_objects(self):
        super(MyLPU, self)._init_objects()
        self.I_ext = parray.to_gpu(np.full(self.num_input, self.I_const,
                                           np.double))

dt = 1e-4
dur = 3.0
steps = int(dur/dt)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=steps, type=int,
                    help='Number of steps [default: %s]' % steps)
parser.add_argument('-g', '--gpu_dev', default=0, type=int,
                    help='GPU device number [default: 0]')
parser.add_argument('-n', type=int, nargs=3, default=(30, 30, 30),
                    help='Numbers of sensory, local, and projection neurons')
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name=file_name, screen=screen)

man = core.Manager()

lpu_name = 'nk'
g = create_lpu_graph(lpu_name, *args.n)
n_dict, s_dict = LPU.graph_to_dicts(g)
total_neurons =  \
    len([d for n, d in g.nodes(data=True) if d['model'] == 'LeakyIAF'])
total_synapses = \
    len([d for f, t, d in g.edges(data=True) if d['model'] == 'AlphaSynapse'])

output_file = 'nk_output.h5'

man.add(MyLPU, lpu_name, dt, n_dict, s_dict, I_const=0.6,
        output_file=output_file,
        device=args.gpu_dev,
        debug=args.debug, time_sync=True)

start = time.time()
man.spawn()
man.start(steps=args.steps)
man.wait()

total_time = time.time()-start
exec_time = man.stop_time-man.start_time

print total_neurons, total_synapses, total_time, exec_time
