#!/usr/bin/env python

import itertools
import os
import time

import numpy as np
from brian2 import *

use_genn = True
use_monitors = True

if use_genn:
    import brian2genn
    set_device('genn')
else:
    prefs.codegen.target = 'weave'
    #prefs.codegen.target = 'numpy'

np.random.seed(0)

N_sensory = 70
N_local = 70
N_proj = 70
N = N_sensory+N_local+N_proj

idx_sensory = range(0, N_sensory)
idx_local = range(N_sensory, N_sensory+N_local)
idx_proj = range(N_sensory+N_local, N_sensory+N_local+N_proj)

Vt = -25.14*mV
Vr = -67.55*mV
R = 1.0244*ohm
C = 66.98*mfarad
tau = R*C

ar = 110.0
ad = 1900.0
gmax = 3*msiemens
E_syn = -65*mV

I_max = 0.6*amp

clock_dt = 0.1*ms
dur = 1.0*second

defaultclock = Clock(dt=clock_dt)

n_model = Equations("""
dV/dt = (R*(I_in+gmax*z*(V-E_syn))-V)/tau : volt
dy/dt = (-(ar+ad)*y-ar*ad*z)/second : 1
dz/dt = y/second : 1
I_in : amp
""")

g = NeuronGroup(N=N, model=n_model, threshold='V>Vt',
                reset='V=Vr')
g.V = np.random.uniform(-0.06, -0.025, N)*volt
g.I_in[0:N_sensory] = I_max

s = Synapses(g, pre='y += ar*ad')
s.z = s.y = '0.0'
tmp = [t for t in itertools.product(idx_sensory, idx_local) 
       if np.random.rand() > 0.5]+\
      [t for t in itertools.product(idx_sensory, idx_proj)
       if np.random.rand() > 0.1]+\
      [t for t in itertools.product(idx_local, idx_proj)
       if np.random.rand() > 0.1]+\
      [t for t in itertools.product(idx_proj, idx_local)
       if np.random.rand() > 0.3]
s.connect([t[0] for t in tmp],
          [t[1] for t in tmp], p=0.5)

if use_monitors:
    spike_mon = SpikeMonitor(g)
    state_mon = StateMonitor(g, True, record=True)

if use_genn:
    dir_name = 'brian2_iaf_alpha_network'
    start = time.time()
    run(dur)
    device.build(directory=dir_name,
                 compile=True,
                 run=True,
                 use_GPU=True)
    total_time = time.time()-start
    with open(os.path.join(dir_name,
        'test_output/test.time'), 'r') as f:
        exec_time =  f.read().strip()
else:
    start = time.time()
    run(dur, profile=True)
    total_time = time.time()-start
    exec_time = sum([t[1] for t in magic_network.profiling_info])

print 'total time: ', total_time
print 'exec time:  ', exec_time

import sys
sys.exit(0)

if not use_monitors:
    import sys
    sys.exit(0)

clf()
subplot(411)
plot(spike_mon.t/ms, spike_mon.i, '.k')
grid(True)
ylim((-0.5, N-0.5))
ya = gca().get_yaxis()
ya.set_major_locator(plt.MaxNLocator(integer=True))
xlim((0, dur/ms))

subplot(412)
for i in xrange(N):
    plot(state_mon.t/ms, state_mon.V[i]/mV)
    hold(True)
    grid(True)
xlim((0, dur/ms))
tight_layout()

subplot(413)
for i in xrange(N):
    plot(state_mon.t/ms, state_mon.y[i])
    hold(True)
    grid(True)
xlim((0, dur/ms))
tight_layout()

subplot(414)
for i in xrange(N):
    plot(state_mon.t/ms, state_mon.z[i])
    hold(True)
    grid(True)
xlim((0, dur/ms))
tight_layout()
