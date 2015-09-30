# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings

import setuptools
import distutils.ccompiler
from distutils.core import Command

from Cython.Distutils import build_ext as _build_ext


__package__ = 'pydriver'

# set to True to skip automatic PCL_HELPER compilation (in this case you have to compile it manually before invoking setup.py)
SKIP_PCL_HELPER = False
if platform.system() == 'Windows':
    # requires manual compilation on Windows
    SKIP_PCL_HELPER = True

# current working directory (directory of setup.py)
cwd = os.path.abspath(os.path.dirname(__file__))

# pcl_helper directories
pcl_helper_dir = os.path.join(__package__, 'pcl', 'pcl_helper')
pcl_helper_dir_build = os.path.join(pcl_helper_dir, 'build')
pcl_helper_dir_lib = os.path.join(pcl_helper_dir, 'lib')

# version.py file path
version_py_path = os.path.join(cwd, __package__, 'version.py')
# source code template for version.py
version_py_src = """# this file was created automatically by setup.py on {timestamp}
__version__ = '{version}'
__version_info__ = {{
    'full': __version__,
    'short': '.'.join(__version__.split('.')[:2])
}}
"""

def read(fname):
    return open(os.path.join(cwd, fname)).read()

def update_version_py():
    """Update version.py using "git describe" command"""
    if not os.path.isdir('.git'):
        print('This does not appear to be a Git repository, leaving version.py unchanged.')
        return False
    try:
        describe_output = subprocess.check_output(['git', 'describe', '--long', '--dirty']).decode('ascii').strip()
    except:
        print('Unable to run Git, leaving version.py unchanged.')
        return False
    # output looks like <version tag>-<commits since tag>-g<hash> and can end with '-dirty', e.g. v0.1.0-14-gd9f10e2-dirty
    # our version tags look like 'v0.1.0' or 'v0.1' and optionally additional segments (e.g. v0.1.0rc1), see PEP 0440
    describe_parts = re.match('^v([0-9]+\.[0-9]+(?:\.[0-9]+)?\S*)-([0-9]+)-g([0-9a-f]+)(?:-(dirty))?$', describe_output)
    assert describe_parts is not None, 'Unexpected output from "git describe": {}'.format(describe_output)
    version_tag, n_commits, commit_hash, dirty_flag = describe_parts.groups()
    n_commits = int(n_commits)
    if dirty_flag is not None:
        print('WARNING: Uncommitted changes detected.')
    if n_commits > 0:
        # non-exact match, dev version
        dev_release = '.dev{}+{}'.format(n_commits, commit_hash)
    else:
        dev_release = ''
    # final version string
    version = version_tag + dev_release
    with open(version_py_path, 'w') as f:
        f.write(version_py_src.format(version=version, timestamp=datetime.datetime.now()))
    print('Set version to: {}'.format(version))
    # success
    return True

# update version.py (if we're in a Git repository)
update_version_py()
# "import" version information without importing the package
exec(open(version_py_path).read())

class build_pcl_helper(Command):
    description = 'build pcl_helper library (inplace)'
    user_options = []
    def initialize_options(self):
        self.cwd_pcl_helper_dir_build = None
    def finalize_options(self):
        # build inplace
        self.cwd_pcl_helper_dir_build = os.path.join(cwd, pcl_helper_dir_build)
    def run(self):
        # create build dir if it doesn't exist
        if not os.path.exists(self.cwd_pcl_helper_dir_build):
            os.makedirs(self.cwd_pcl_helper_dir_build)
        # build pcl_helper
        if platform.system() == 'Windows':
            self._build_pcl_helper_windows(self.cwd_pcl_helper_dir_build)
        else:
            self._build_pcl_helper_linux(self.cwd_pcl_helper_dir_build)
    def _build_pcl_helper_linux(self, build_dir):
        subprocess.check_call(['cmake', '..'], cwd=build_dir)
        subprocess.check_call('make', cwd=build_dir)
    def _build_pcl_helper_windows(self, build_dir):
        raise NotImplementedError

class build_ext(_build_ext):
    user_options = _build_ext.user_options + [
        ('skip-pcl-helper', None, 'skip pcl_helper compilation (assume manual compilation)'),
    ]
    boolean_options = _build_ext.boolean_options + ['skip-pcl-helper']

    def initialize_options(self):
        _build_ext.initialize_options(self)
        # don't skip pcl helper by default
        self.skip_pcl_helper = False
        # pcl_helper location in source directory
        self.cwd_pcl_helper_dir_lib = None
        # pcl_helper location in package build directory
        self.build_pcl_helper_dir_lib = None

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy as np
        self.include_dirs.append(np.get_include())
        # finalize pcl_helper directories
        self.cwd_pcl_helper_dir_lib = os.path.join(cwd, pcl_helper_dir_lib)
        self.build_pcl_helper_dir_lib = os.path.join(self.build_lib, pcl_helper_dir_lib)
        # check global flag SKIP_PCL_HELPER
        self.skip_pcl_helper = self.skip_pcl_helper or SKIP_PCL_HELPER

    def build_extensions(self, *args, **kwargs):
        compiler_type = self.compiler.compiler_type
        if compiler_type not in extra_args:
            compiler_type = 'unix'  # probably some unix-like compiler
        # merge compile and link arguments with global arguments for current compiler
        for e in self.extensions:
            e.extra_compile_args = list(set(e.extra_compile_args + extra_args[compiler_type]['extra_compile_args']))
            e.extra_link_args = list(set(e.extra_link_args + extra_args[compiler_type]['extra_link_args']))
        _build_ext.build_extensions(self, *args, **kwargs)

    def run(self):
        if not self.skip_pcl_helper:
            # build pcl_helper first
            try:
                self.run_command('build_pcl_helper')
            except:
                print('Error: pcl_helper could not be compiled automatically')
                print('Please compile pcl_helper manually (see %s/pcl/pcl_helper/README.rst for instructions)' % __package__ + \
                    ' and set SKIP_PCL_HELPER in setup.py to True.')
                raise
        # copy pcl_helper library to package build directory
        self.copy_tree(self.cwd_pcl_helper_dir_lib, self.build_pcl_helper_dir_lib)
        _build_ext.run(self)

    def get_outputs(self):
        # add contents of pcl_helper library directory to outputs (so they can be uninstalled)
        outputs = []
        for dirpath, dirnames, filenames in os.walk(self.build_pcl_helper_dir_lib):
            outputs.extend([os.path.join(dirpath, f) for f in filenames])
        return _build_ext.get_outputs(self) + outputs

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        self._remove_dirs('__pycache__')

        self._remove_dir(cwd, 'build')
        self._remove_dir(cwd, 'build_c')
        self._remove_dir(cwd, 'dist')
        self._remove_dir(cwd, '.eggs')
        self._remove_dir(cwd, '{}.egg-info'.format(__package__))
        self._remove_dir(cwd, pcl_helper_dir_build)
        self._remove_dir(cwd, pcl_helper_dir_lib)

        self._remove_files('pyc')
        self._remove_files('pyo')
        self._remove_files('pyd')
        self._remove_files('so')
    def _remove_dirs(self, dirname, parent_dir=None):
        if parent_dir is None:
            full_parent_dir = cwd
        else:
            full_parent_dir = os.path.join(cwd, parent_dir)
        matches = []
        for dirpath, dirnames, filenames in os.walk(full_parent_dir):
            matches.extend([os.path.join(dirpath, d) for d in dirnames if d==dirname])
        for d in matches:
            self._remove_dir(d)
    def _remove_dir(self, *args):
        dirpath = os.path.abspath(os.path.join(*args))
        # sanity checks
        if not os.path.exists(dirpath):
            # nothing to do
            return
        if not os.path.isdir(dirpath):
            print('"{}" is not a directory, aborting...'.format(dirpath))
            sys.exit()
        path_check = True
        if not dirpath.startswith(cwd):
            path_check = False
        if path_check and len(dirpath) > len(cwd):
            # first character after cwd should be a slash or a backslash
            if dirpath[len(cwd)] != os.sep:
                path_check = False
        if not path_check:
            print('The directory "{}" appears to be outside of main directory ({}), aborting...'.format(dirpath, cwd))
            sys.exit()
        # all sanity checks ok
        if not os.path.islink(dirpath):
            print('Removing directory: ' + dirpath)
            shutil.rmtree(dirpath, ignore_errors=True)
        else:
            print("Can't remove symlink to directory: " + dirpath)
    def _remove_files(self, ext, parent_dir=None):
        if parent_dir is None:
            full_parent_dir = cwd
        else:
            full_parent_dir = os.path.join(cwd, parent_dir)
        matches = []
        for dirpath, dirnames, filenames in os.walk(full_parent_dir):
            matches.extend([os.path.join(dirpath, f) for f in filenames if f.endswith('.'+ext)])
        for f in matches:
            self._remove_file(f)
    def _remove_file(self, *args):
        filepath = os.path.abspath(os.path.join(*args))
        # sanity checks
        if not os.path.exists(filepath):
            # nothing to do
            return
        if not os.path.isfile(filepath):
            print('"{}" is not a file, aborting...'.format(filepath))
            sys.exit()
        filepath_dir = os.path.abspath(os.path.dirname(filepath))
        path_check = True
        if not filepath_dir.startswith(cwd):
            path_check = False
        if path_check and len(filepath_dir) > len(cwd):
            # first character after cwd should be a slash or a backslash
            if filepath_dir[len(cwd)] != os.sep:
                path_check = False
        if not path_check:
            print('The file "{}" appears to be outside of main directory ({}), aborting...'.format(filepath, cwd))
            sys.exit()
        # all sanity checks ok
        print('Removing file: ' + filepath)
        os.remove(filepath)

class lazy_cythonize(list):
    # cythonize only if needed (e.g. not for "clean" command)
    def __init__(self, extensions, *args, **kwargs):
        self._list = None
        self.extensions = extensions
        self.args = args
        self.kwargs = kwargs
    def c_list(self):
        if self._list is None:
            self._list = self._cythonize()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())
    def _cythonize(self):
        from Cython.Build import cythonize
        return cythonize(self.extensions, *self.args, **self.kwargs)

# setup argument parser
parser = argparse.ArgumentParser(
    description = '%s setup script, basic install: python setup.py install' % __package__,
    epilog = 'Other arguments will be passed to setuptools, use --help-setup for more information.',
)
# add arguments which we will parse and pass to setuptools
parser.add_argument('command', nargs = '?',
                    help = 'command to pass to setuptools, use "install" to install package')
parser.add_argument('--debug', '-g', action = 'store_true',
                    help = 'compile/link with debugging information')
parser.add_argument('--force', '-f', action = 'store_true',
                    help = 'forcibly build everything (ignore file timestamps)')
parser.add_argument('--help-setup', action = 'store_true',
                    help = 'show setuptools help and exit')
# add own arguments
parser.add_argument('--annotate', action = 'store_true',
                    help = 'let Cython generate HTML files with performance information')
parser.add_argument('--cython-build-dir', default = 'build_c',
                    help = 'directory for C/C++ sources and HTML files generated by Cython (default: build_c)')
parser.add_argument('--inplace', action = 'store_true',
                    help = 'build inplace')
parser.add_argument('--no-openmp', dest = 'openmp', action = 'store_false',
                    help = 'compile/link without OpenMP support')
parser.add_argument('--profile', action = 'store_true',
                    help = 'enable profiling with cProfile')
parser.add_argument('--skip-pcl-helper', action = 'store_true',
                    help = 'skip pcl_helper compilation (assume manual compilation)')

# parse command line arguments
cmdargs, unknown_args = parser.parse_known_args()

if cmdargs.help_setup:
    # show setuptools help and exit
    sys.argv = [sys.argv[0], '--help']
    setuptools.setup()
    sys.exit()

# construct new command line arguments for setuptools
# leave the script name
setuptools_argv = [sys.argv[0]]
# pass setuptools options which we already have parsed
if cmdargs.command: setuptools_argv.append(cmdargs.command)
if cmdargs.force: setuptools_argv.append('--force')
if cmdargs.debug: setuptools_argv.append('--debug')
# add all unknown args
setuptools_argv += unknown_args
# replace sys.argv by arguments which will be passed to setuptools
sys.argv = setuptools_argv

# initialize setuptools arguments
setup_args = {
    'name': __package__,
    'version': __version__,
    'url': 'http://github.com/lpltk/pydriver',
    'license': 'MIT',
    'author': 'Leonard Plotkin',
    'author_email': 'git@leonard-plotkin.de',
    'description': 'A framework for training and evaluating object detectors and classifiers in road traffic environment.',
    'long_description': read('README.rst'),
    'zip_safe': False,
    'package_dir': {__package__: __package__},
    'packages': setuptools.find_packages(),
    'package_data': {__package__+'.pcl': ['pcl_helper/lib/*']},
    'include_package_data': True,
    'platforms': 'any',
    'setup_requires': [
        'numpy>=1.8.1',
        'cython>=0.22.1',
    ],
    'install_requires': [
        'numpy>=1.8.1',
        'cython>=0.22.1',
        'scipy>=0.13.3',
        'scikit-image',
        'scikit-learn',
        'shapely',
    ],
    'classifiers': [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
}
# common include directories
setup_args['include_dirs'] = [
    os.path.join(__package__, 'common'),    # common constants, structs and typedefs
]

extra_args = {}
# arguments for unix compilers
extra_args['unix'] = {'extra_compile_args': [], 'extra_link_args': []}
if cmdargs.openmp:
    extra_args['unix']['extra_compile_args'].append('-fopenmp')
    extra_args['unix']['extra_link_args'].append('-fopenmp')
if not cmdargs.debug:
    extra_args['unix']['extra_compile_args'].append('-O3')  # maximum optimization
    extra_args['unix']['extra_compile_args'].append('-w')   # suppress warnings
# arguments for MSVC
# PYDs compiled with MSVC and /openmp could not be embedded in a MSVC project (regardless of the project's /openmp setting).
# They crash with the error R6034: An application has made an attempt to load the C runtime library incorrectly.
# The problem seems to be that the resulting exe and pyd binaries had incompatible manifests. It could also be caused by using
# the Visual Studio Express Edition which has limited support for extended features such as OpenMP.
extra_args['msvc'] = {'extra_compile_args': [], 'extra_link_args': []}
extra_args['msvc']['extra_compile_args'].append('/EHsc')    # exception handling option
if cmdargs.openmp:
    extra_args['msvc']['extra_compile_args'].append('/openmp')
if not cmdargs.debug:
    extra_args['msvc']['extra_compile_args'].append('/O2')  # optimize for speed
    extra_args['msvc']['extra_compile_args'].append('/W0')  # suppress warnings

# helper function for creating extensions with standard options
def create_extension(*args, **kwargs):
    def add_package_path(kwarg_key):
        # prepend package directory to every element in list in kwargs[kwarg_key]
        kwargs[kwarg_key] = [os.path.join(__package__, e) for e in kwargs.get(kwarg_key, [])]

    # add package name to extension module name and paths
    args = (__package__ + '.' + args[0],) + args[1:]
    add_package_path('sources')
    add_package_path('include_dirs')
    add_package_path('library_dirs')
    # generate C++ code (instead of C) by default
    if 'language' not in kwargs:
        kwargs['language'] = 'c++'
    return setuptools.Extension(*args, **kwargs)

extensions = [
        # common
        create_extension(
            'common.constants',
            sources = ['common/constants.pyx'],
        ),
        create_extension(
            'common.functions',
            sources = ['common/functions.pyx'],
        ),

        # geometry
        create_extension(
            'geometry.geometry',
            sources = ['geometry/geometry.pyx'],
        ),

        # stereo
        create_extension(
            'stereo.stereo',
            sources = ['stereo/stereo.pyx'],
        ),

        # pcl
        create_extension(
            'pcl.pcl',
            sources = ['pcl/pcl.pyx'],
            language = 'c++',
            include_dirs = ['pcl/pcl_helper'],
            libraries = ['pcl_helper'],
            library_dirs = ['pcl/pcl_helper/lib'],
            extra_link_args = ['-Wl,-rpath,$ORIGIN/pcl_helper/lib'] if platform.system() != 'Windows' else [], # handle paths in __init__.py on Windows
        ),

        # preprocessing
        create_extension(
            'preprocessing.preprocessing',
            sources = ['preprocessing/preprocessing.pyx'],
        ),

        # keypoints
        create_extension(
            'keypoints.base',
            sources = ['keypoints/base.pyx'],
        ),
        create_extension(
            'keypoints.harris',
            sources = ['keypoints/harris.pyx'],
        ),
        create_extension(
            'keypoints.iss',
            sources = ['keypoints/iss.pyx'],
        ),

        # features
        create_extension(
            'features.shot',
            sources = ['features/shot.pyx'],
        ),
        create_extension(
            'features.base',
            sources = ['features/base.pyx'],
        ),

        # detectors
        create_extension(
            'detectors.vocabularies',
            sources = ['detectors/vocabularies.pyx'],
        ),
        create_extension(
            'detectors.detectors',
            sources = ['detectors/detectors.pyx'],
        ),
    ]

# setup commands
setup_args['cmdclass'] = {
    'build_pcl_helper': build_pcl_helper,
    'build_ext': build_ext,
    'clean': CleanCommand,
}
# keyword arguments for cythonize()
cython_kwargs = {
    'build_dir': cmdargs.cython_build_dir,  # build directory
    'compiler_directives': {
        'embedsignature': True,             # embed signatures for documentation tools
    },
}
if cmdargs.force:
    cython_kwargs['force'] = True           # enforce full recompilation
if cmdargs.annotate:
    cython_kwargs['annotate'] = True        # generate HTML reports (in cmdargs.cython_build_dir)
if cmdargs.profile:
    if cmdargs.debug:
        warnings.warn(UserWarning('You should only profile in release mode with full optimization.'))
    cython_kwargs['profile'] = True         # globally enable profiling with cProfile
setup_args['ext_modules'] = lazy_cythonize(extensions, **cython_kwargs)
setup_args['options'] = {
    'build_ext': {
        'inplace': cmdargs.inplace,
        'skip_pcl_helper': cmdargs.skip_pcl_helper,
    },
}

try:
    setuptools.setup(**setup_args)
except Exception as e:
    print('Compilation errors encountered, aborting...')
    print('Exception information:')
    print(e)
    sys.exit(1)
