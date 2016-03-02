#!/usr/bin/env python

"""
Generic LPU demo

Notes
-----
Generate input file and LPU configuration by running

cd data
python gen_generic_lpu.py
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
    def __init__(self, dt, n_dict, s_dict, one_time_import=10, input_file=None, output_file=None,
                 device=0, ctrl_tag=core.CTRL_TAG, gpot_tag=core.GPOT_TAG,
                 spike_tag=core.SPIKE_TAG, rank_to_id=None, routing_table=None,
                 id=None, debug=False, columns=['io', 'type', 'interface'],
                 cuda_verbose=False, time_sync=False):
        super(MyLPU, self).__init__(dt, n_dict, s_dict, input_file, output_file,
                 device, ctrl_tag, gpot_tag,
                 spike_tag, rank_to_id, routing_table,
                 id, debug, columns,
                 cuda_verbose, time_sync)

        # Force all data to be loaded into memory in one operation:
        self.one_time_import = one_time_import

output_file = None # 'nk_output.h5'
input_file = 'nk_input.h5'
import h5py
f = h5py.File(input_file)
one_time_import = f['/array'].shape[0]
f.close()

man.add(MyLPU, 'nk', dt, n_dict, s_dict, one_time_import,
        input_file='nk_input.h5',
        output_file=None,
        device=args.gpu_dev,
        debug=args.debug, time_sync=True)

start = time.time()
man.spawn()
man.start(steps=args.steps)
man.wait()

total_time = time.time()-start
exec_time = man.stop_time-man.start_time

print total_synapses, total_time, exec_time
