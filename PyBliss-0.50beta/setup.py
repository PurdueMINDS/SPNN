from distutils.core import setup, Extension

#
# The directory in which the true bliss is hiding
#
blissdir = './bliss-0.50'
# The essential bliss source files
blisssrcs = ['graph.cc','heap.cc','orbit.cc','partition.cc',
             'timer.cc','uintseqhash.cc']
blisssrcs = [blissdir+'/'+src for src in blisssrcs]

module1 = Extension('intpybliss',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '50beta')],
                    include_dirs = [blissdir],
                    sources = ['intpyblissmodule.cc']+blisssrcs
                    )

setup (name = 'IntPyBliss',
       version = '0.50beta',
       description = 'This is an internal PyBliss package',
       author = 'Tommi Junttila',
       #author_email = '',
       #url = 'http://www.python.org/doc/current/ext/building.html',
       long_description = '''
This is an internal PyBliss package.
Should never be used directly.
''',
       ext_modules = [module1])
