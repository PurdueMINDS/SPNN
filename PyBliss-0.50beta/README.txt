PyBliss - a Python wrapper for bliss
(c) 2008-2010 Tommi Junttila

PyBliss is a Python (http://www.python.org/) wrapper for the
bliss graph canonical labeling tool (http://www.tcs.hut.fi/Software/bliss/).
The source code of bliss is included in this directory.
For performance critical software, please use the C++ interface of bliss
instead of this Python wrapper.

To compile PyBliss on Linux platforms, just say
  make
and hope that everything goes smoothly.
To compile on non-Linux platforms, you're pretty much on your own and
should perhaps consult
  http://docs.python.org/install/index.html

After succesfull compilation, you can type
  python test_enumerate.py 
and should see an output ending in the lines:
  There are 4 non-isomorphic graphs with 3 vertices
  There are 34 non-isomorphic graphs with 5 vertices
  There are 156 non-isomorphic graphs with 6 vertices

Assume that PyBliss (including this file) is in a directory <PYBLISSDIR>.
To import and use PyBliss from other python modules, use:
  import sys
  sys.path.append('<PYBLISSDIR>')
  sys.path.append('<PYBLISSDIR>/lib/python')
  import PyBliss
After this, you can use
  help(PyBliss)
See test_enumerate.py for an example of how to use PyBliss.
