============
Installation
============

Although the PyDriver framework was only tested with Ubuntu 14.04 and Windows 7 so far, it should
be usable with every operating system which supports Python 2.7 and all required components.

The `PyDriver repository <https://github.com/lpltk/pydriver>`_ is located at GitHub.

**Required software**

- `Python <https://www.python.org/>`_ (2.7)
- `CMake <http://www.cmake.org/>`_
- `Point Cloud Library (PCL) <http://pointclouds.org/>`_ (>=1.7.1)

**Required Python packages**

- `pip <https://pypi.python.org/pypi/pip>`_
- `Cython <http://cython.org/>`_ (>=0.22.1)
- `SciPy <http://www.scipy.org/>`_ (NumPy>=1.8.1, SciPy Library, Matplotlib)
- `scikit-learn <http://scikit-learn.org/>`_
- `scikit-image <http://scikit-image.org/>`_
- `Shapely <https://pypi.python.org/pypi/Shapely>`_

**Optional software**

- `OpenCV <http://opencv.org/>`_ (used for stereo image processing)
- `PyOpenCL <http://documen.tician.de/pyopencl/>`_ and hardware drivers with OpenCL support (enables GPU usage)
- `Sphinx <http://sphinx-doc.org/>`_ (documentation generator)

------------
Ubuntu 14.04
------------

Python 2.7 and the GCC compiler are included by default in Ubuntu 14.04.

PCL installation

.. code-block:: none

    sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
    sudo apt-get update
    sudo apt-get install libpcl-all

Other required packages

.. code-block:: none

    sudo apt-get install cmake python-dev python-pip python-scipy python-skimage python-shapely
    sudo pip install cython sklearn --upgrade

Optional packages

.. code-block:: none

    sudo apt-get install python-opencv
    sudo apt-get install python-pyopencl
    sudo pip install sphinx

Switch to the PyDriver source directory (the one with setup.py in it) and install the package.

.. code-block:: none

    sudo pip install .

Now you can change your working directory to something else (so Python won't import the uncompiled
source code, that will result in an error), run the Python interpreter and try *import pydriver*.


-------
Windows
-------

You have multiple options for installing PyDriver on Windows depending on your needs. You can
download compiled binaries from https://github.com/lpltk/pydriver/releases. They are currently
available for x64 systems only.

Standalone archive
------------------
This archive is a WinPython distribution with pre-installed PyDriver package. Extract it and
run the WinPython command prompt. You should be able to start Python there and execute
*import pydriver*.

MSI package installer
---------------------
Use the package installer to install PyDriver in an existing Python installation. You will need
to install the required Python packages manually, but this option does not require Point Cloud
Library or CMake to be installed. See `Compile from source`_ for more information about installing
dependencies.

Python binary wheel
-------------------
You can also use a binary wheel to install PyDriver in an existing Python installation. The installation
command is *pip install <wheel.whl>*. The required Python packages will be installed automatically, but
there are known issues with some of them. Specifically, you should install Shapely from its binary wheel
manually instead of relying on automatic installation. This option does not require Point Cloud
Library or CMake to be installed. See `Compile from source`_ for more information about installing
dependencies.

Compile from source
-------------------

For Windows the recommended way to install large parts of the required software is to use
`WinPython <https://winpython.github.io/>`_ that already includes Python, Cython, NumPy, SciPy,
Matplotlib and other packages. The homepage of `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_,
who is doing great work maintaining it, offers binary wheels for many Python packages including
those for which an official Windows binary distribution is not provided. Keep in mind that you will probably
need large amounts of training data for sensible results and therefore you should use 64-bit packages.

The PCL installer for Windows is currently (August 2015) outdated and PCL must be compiled from source.
Versions prior to 1.7.1 are not compatible with PyDriver. The recommended compiler for PCL 1.7.1 on
Windows is Visual C++ 2010 (i.e. 10.0). After installing PCL the pcl_helper library (in *pcl/pcl_helper*)
must be compiled with the same compiler used for PCL. You have to generate a Visual C++ project in
*pcl_helper/build* with CMake and supplied CMake configuration files. Now you can build the generated
project with Visual Studio. Remember that you may want to switch to the "Release" configuration.

To compile Cython extensions for Python you should use the same compiler which was used to compile
Python. For the standard Python 2.7 Windows distribution it's Visual C++ 2008 (i.e. 9.0). Your compiler
version must support 64-bit binaries in order to use 64-bit Python packages. Support for OpenMP is optional
and will allow Cython code to make use of multiple CPU cores. The recommended way is to use the
`Microsoft Visual C++ Compiler for Python 2.7 <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`_ (9.0).
See its installation instructions for dependencies which should be installed first. Remember to install Visual
C++ compilers in the order of their versions.

The final step is to switch to the PyDriver source directory (the one with setup.py in it), compile and
install it (administrator privileges may be required):

.. code-block:: none

    python setup.py build_ext --compiler=msvc
    pip install .

.. note::
    WinPython and generally most Python packages are portable in the sense that they can be run without
    installation, e.g. from a USB stick. If you compile PCL as a static library and compile the pcl_helper
    library against it, the pcl_helper library will be portable in the same way. Thus you can make a completely
    portable PyDriver package. However, you won't be able to modify the pcl_helper library without installing PCL.


----------
Developers
----------

See "Makefile" in the PyDriver source code repository for additional options like installing in
editable mode, generating documentation and other useful commands.
