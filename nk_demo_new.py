#!/usr/bin/env python

"""
Generic LPU demo

Notes
-----
Generate input file and LPU configuration by running

python gen_nk_lpu.py
"""

import argparse
import itertools
import time

import networkx as nx

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU

import neurokernel.mpi_relaunch

dt = 1e-4
dur = 5.0
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
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name=file_name, screen=screen)

man = core.Manager()

lpu_file = 'nk_lpu.gexf.gz'
(n_dict, s_dict) = LPU.lpu_parser(lpu_file)
g = nx.read_gexf(lpu_file)
total_synapses = \
    len([d for f, t, d in g.edges(data=True) if d['model'] == 'AlphaSynapse'])

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
            self.output_gpot_buffer.append(
                self.V.get()[self.gpot_order_l].reshape((-1,)))
        if self.total_num_spike_neurons > 0:
            self.output_spike_buffer.append(
                self.spike_state.get()[self.spike_order_l].reshape((-1,)))

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

    def _init_objects(self):
        super(MyLPU, self)._init_objects()
        self.I_ext = parray.to_gpu(np.full(self.num_input, self.I_const,
                                           np.double))

output_file = None # 'nk_output.h5'

man.add(MyLPU, 'nk', dt, n_dict, s_dict, I_const=0.6,
        output_file=output_file,
        device=args.gpu_dev,
        debug=args.debug, time_sync=True)

start = time.time()
man.spawn()
man.start(steps=args.steps)
man.wait()

total_time = time.time()-start
exec_time = man.stop_time-man.start_time

print total_synapses, total_time, exec_time
