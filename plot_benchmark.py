#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import path as path

p = path.Path('.')
columns = ['neurons', 'synapses', 'total_time', 'exec_time']
df_nd_list = [pd.read_csv(f, sep=' ', header=None, names=columns)
                for f in p.glob('neurodriver*log')]
df_bg_list = [pd.read_csv(f, sep=' ', header=None, names=columns)
                for f in p.glob('brian2genn*log')]
assert len(df_nd_list) == len(df_bg_list)
M = len(df_nd_list)

N = len(df_nd_list[0])
df_nd_avg = pd.DataFrame(0, index=range(N), columns=columns)
df_bg_avg = pd.DataFrame(0, index=range(N), columns=columns)

df_nd_avg['neurons'] = df_nd_list[0]['neurons']
df_nd_avg['synapses'] = df_nd_list[0]['synapses']
df_nd_avg['total_time'] = sum([df['total_time'] for df in df_nd_list])/M
df_nd_avg['exec_time'] = sum([df['exec_time'] for df in df_nd_list])/M

df_bg_avg['neurons'] = df_bg_list[0]['neurons']
df_bg_avg['synapses'] = df_bg_list[0]['synapses']
df_bg_avg['total_time'] = sum([df['total_time'] for df in df_bg_list])/M
df_bg_avg['exec_time'] = sum([df['exec_time'] for df in df_bg_list])/M

# Plot exec_time for Neurodriver and total_time for Brian2GeNN because
# the exec_time for Brian2GeNN contains the timings for the generated 
# binary only without including the time taken for compilation and
# transmitting generated data back into Python:
plt.clf()
plt.subplot(211)
plt.plot(df_nd_avg['neurons'], df_nd_avg['exec_time'], 'b',
         df_bg_avg['neurons'], df_bg_avg['total_time'], 'r')
plt.xlabel('# of neurons')
plt.ylabel('t (s)')
plt.grid(True)
plt.axis('tight')
curr_ylim = plt.ylim()
plt.ylim((curr_ylim[0]*0.9, curr_ylim[1]*1.2))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.title('Execution Duration for 3.0 s Emulation at $\Delta t$ = 1e-4')
plt.legend(['Neurodriver', 'Brian2GeNN'], loc='upper left')

plt.subplot(212)
plt.plot(df_nd_avg['synapses'], df_nd_avg['exec_time'], 'b',
         df_bg_avg['synapses'], df_bg_avg['total_time'], 'r')
plt.xlabel('# of synapses')
plt.ylabel('t (s)')
plt.grid(True)
plt.axis('tight')
curr_ylim = plt.ylim()
plt.ylim((curr_ylim[0]*0.9, curr_ylim[1]*1.2))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('neurodriver-benchmark.png', dpi=120)
