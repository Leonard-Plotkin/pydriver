============
Installation
============

Although the PyDriver framework was only tested with Debian 8 and Windows 7 so far, it should
be usable with every operating system which supports Python 2.7 or 3.x and all required components.

The `PyDriver repository <https://github.com/lpltk/pydriver>`_ is located at GitHub.

**Required software**

- `Python <https://www.python.org/>`_ (2.7 or 3.x)
- `CMake <http://www.cmake.org/>`_
- `Point Cloud Library (PCL) <http://pointclouds.org/>`_ (>=1.7.1)

**Required Python packages**

- `pip <https://pypi.python.org/pypi/pip>`_
- `Cython <http://cython.org/>`_ (>=0.22.1)
- `SciPy <http://www.scipy.org/>`_ (NumPy>=1.8.1, SciPy Library, Matplotlib)
- `scikit-learn <http://scikit-learn.org/>`_
- `scikit-image <http://scikit-image.org/>`_

**Optional software**

- `OpenCV <http://opencv.org/>`_ (used for stereo image processing)
- `PyOpenCL <http://documen.tician.de/pyopencl/>`_ and hardware drivers with OpenCL support (enables GPU usage)
- `Shapely <https://pypi.python.org/pypi/Shapely>`_ (enables non-maximum suppression during object recognition)
- `Sphinx <http://sphinx-doc.org/>`_ (documentation generator)

--------
Debian 8
--------

**PCL installation**

.. code-block:: none

    sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
    sudo apt-get update
    sudo apt-get install libpcl-all

**CMake**

.. code-block:: none

    sudo apt-get install cmake

Python 3.x
----------

**Required Python packages**

.. code-block:: none

    sudo apt-get install python3-dev python3-scipy python3-skimage cython3
    sudo pip3 install --upgrade cython sklearn

**Optional Python packages**

OpenCV 3 has support for Python 3.x, but manual installation may be required.

.. code-block:: none

    sudo apt-get install python3-pyopencl python3-shapely
    pip3 install --upgrade sphinx

Python 2.7
----------

**Required Python packages**

.. code-block:: none

    sudo apt-get install python-dev python-pip python-scipy python-skimage cython
    sudo pip2 install --upgrade cython sklearn

**Optional Python packages**

.. code-block:: none

    sudo apt-get install python-opencv python-shapely
    sudo apt-get install python-pyopencl
    pip2 install --upgrade sphinx

Final step
----------

**Python 3.x**

.. code-block:: none

    pip3 install pydriver

**Python 2.7**

.. code-block:: none

    pip2 install pydriver

Now you can run the Python interpreter and try *import pydriver*.


-------
Windows
-------

You have multiple options for installing PyDriver on Windows depending on your needs. You can
download compiled binaries from https://github.com/lpltk/pydriver/releases. They are currently
available for x64 systems only.

You may need to install Microsoft Visual C++ 2010 redistributable package if it's not already
installed on your system. For Python 3.5 and later you will also need the 2015 version whereas for
Python 3.2 and earlier the 2008 version is required.

Standalone archive
------------------
This archive is a WinPython distribution with pre-installed PyDriver package. Extract it and
run the WinPython command prompt. You should be able to start Python there and execute
*import pydriver*.

Python binary wheel
-------------------
Use the binary wheel to install PyDriver in an existing Python installation. The installation
command is *pip install <wheel.whl>*. The required Python packages should be installed
automatically, but this is not always possible. In this case you may need to install them from
their binary wheels manually instead of relying on automatic installation. This option does not
require Point Cloud Library or CMake to be installed. See `Compile from source`_ for more
information about installing dependencies.

MSI package installer
---------------------
You can also use the package installer to install PyDriver in an existing Python installation. You
will need to install the required Python packages manually, but this option does not require Point
Cloud Library or CMake to be installed. See `Compile from source`_ for more information about
installing dependencies.

Compile from source
-------------------

**Dependencies**

For Windows the recommended way to install large parts of the required software is to use
`WinPython <https://winpython.github.io/>`_ that already includes Python, Cython, NumPy, SciPy,
Matplotlib and other packages. The homepage of `Christoph Gohlke
<http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_, who is doing great work maintaining it, offers
binary wheels for many Python packages including those for which an official Windows binary
distribution is not provided. Keep in mind that you will probably need large amounts of memory
and therefore you should use 64-bit Python and corresponding packages.

**PCL**

The PCL installer for Windows is currently (October 2015) outdated and PCL must be compiled from
source. Versions prior to 1.7.1 are not compatible with PyDriver. The recommended compiler for
PCL 1.7.1 on Windows is Visual C++ 2010 (10.0). After installing PCL the pcl_helper library
(in *pcl/pcl_helper*) must be compiled with the same compiler used for PCL. You have to generate
a Visual C++ project in *pcl_helper/build* with CMake and supplied CMake configuration files. Now
you can build the generated project with Visual Studio. Remember that you may want to switch to
the "Release" configuration.

**Compiler**

To compile Cython extensions for Python you should use the same compiler which was used to compile
Python. For the standard Python 3.5 Windows Distribution it's Visual C++ 2015 (14.0), for Python
3.4 and 3.3 it's VC++ 2010 (10.0), and for earlier Python versions it's VC++ 2008 (9.0). You can
also try using the Mingw-w64 compiler included in the latest WinPython distributions and being run
by default. Your compiler version must support 64-bit binaries in order to use 64-bit Python
packages. Support for OpenMP is optional and will allow Cython code to make use of multiple CPU
cores. The recommended way for installing VC++ 2008 is to use the `Microsoft Visual C++ Compiler
for Python 2.7 <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`_ . See its
installation instructions for dependencies which should be installed first. Remember to install
Visual C++ compilers in the order of their versions.

**Final step**

The final step is to switch to the PyDriver source directory (the one with setup.py in it),
compile and install it (administrator privileges may be required):

.. code-block:: none

    pip install .

If this does not succeed, try this instead:

.. code-block:: none

    python setup.py build_ext --compiler=msvc
    pip install .

You may need to run *python setup.py clean* and recompile (or backup) the pcl_helper library
before doing this.

.. note::
    WinPython and generally most Python packages are portable in the sense that they can be run
    without installation, e.g. from a USB stick. If you compile PCL as a static library and
    compile the pcl_helper library against it, the pcl_helper library will be portable in the same
    way. Thus you can make a completely portable PyDriver package. However, you won't be able to
    modify the pcl_helper library without installing PCL.


----------
Developers
----------

See "Makefile" in the PyDriver source code repository for additional options like installing in
editable mode, generating documentation and other useful commands.
