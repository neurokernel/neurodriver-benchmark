#!/usr/bin/env python

import sarge

for n in xrange(50, 2500, 50):
    with sarge.Capture() as out:
        sarge.run('python -u nk_demo.py -n %s %s %s' % (n, n, n),
                  stdout=out)
    print out.read(),
