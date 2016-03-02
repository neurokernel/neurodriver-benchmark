#!/usr/bin/env python

import subprocess

try:
    from subprocess import DEVNULL
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

for n in xrange(50, 1500, 50):
    out = subprocess.check_output(['python', 'brian2genn_demo.py', '-n',
                             str(n), str(n), str(n)], stderr=DEVNULL)
    print out,
