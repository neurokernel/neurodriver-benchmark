#!/usr/bin/env python

import re

import sarge

for n in xrange(50, 4000, 50):
    c = sarge.capture_both('python -u brian2genn_demo.py -n %s %s %s' % (n, n, n))
    for line in c.stdout.readlines():
        if re.match('^\d.*', line):
            print line,
