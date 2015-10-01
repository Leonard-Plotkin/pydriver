What is PCL_HELPER?
===================
PCL_HELPER is a custom Python interface to Point Cloud Library used in
PyDriver. The main installation script will try to take care of its
compilation. In case it doesn't succeed you have to compile PCL_HELPER
manually using these instructions.

CMake
=====
Use CMake to build a project for PCL_HELPER which is expected to be placed
in pcl_helper/build. Choose the generator matching your Point Cloud Library
installation.

Tested with following generators: Unix Makefiles, Visual Studio 10

Build
=====
Run "make" for Unix Makefiles or build the solution in Visual Studio. The
resulting library is expected to be in pcl_helper/lib.

Note
====
You can mix different compilers for PCL_HELPER (must match the one used for
PCL) and PyDriver (must match the one used for Python) if you need to.
