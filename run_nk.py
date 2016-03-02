#!/usr/bin/env python

import subprocess

try:
    from subprocess import DEVNULL
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

for n in xrange(50, 1500, 50):
    subprocess.check_output(['python', 'gen_nk_lpu.py', '-n',
                             str(n), str(n), str(n)], stderr=DEVNULL)
    out = subprocess.check_output(['python', 'nk_demo.py'],
                                  stderr=DEVNULL)
    print out,
