# -*- coding: utf-8 -*-
"""OpenCL configuration"""
from __future__ import absolute_import, division

import warnings

__all__ = ['USE_OPENCL', 'PREFERRED_DEVICE']
# USE_OPENCL: flag whether OpenCL can be used
# PREFERRED_DEVICE: dictionary with device information (or None if OpenCL can't be used), keys:
#    device: Device instance
#    context: Context instance
#    queue: CommandQueue instance


try:
    import pyopencl as cl
    # OpenCL can be used
    USE_OPENCL = True
except ImportError:
    # OpenCL can't be used
    USE_OPENCL = False

# initialize preferred device dictionary to None
PREFERRED_DEVICE = None

if USE_OPENCL:
    # find best device in terms of max_compute_units * max_clock_frequency
    bestDevice = None
    bestPerformance = 0
    try:
        platforms = cl.get_platforms()
    except:
        warnings.warn(UserWarning('Error during OpenCL initialization, get_platforms() failed.'))
        platforms = []

    for platform in platforms:
        try:
            devices = platform.get_devices()
        except:
            warnings.warn(UserWarning('Error during OpenCL initialization, platform.get_devices() failed.'))
            devices = []
        for device in devices:
            if device.available and device.compiler_available and device.linker_available:
                # calculate performance measure
                performance = device.max_compute_units * device.max_clock_frequency
                if performance > bestPerformance:
                    # found better device
                    bestDevice = device
                    bestPerformance = performance

    if bestDevice is None:
        # couldn't find any devices, can't use OpenCL
        USE_OPENCL = False
        warnings.warn(UserWarning('No suitable OpenCL devices found, functionality deactivated.'))
    else:
        # initialize device
        device = bestDevice
        context = cl.Context([bestDevice])
        queue = cl.CommandQueue(context)
        # store initialized object instances in preferred device dictionary
        PREFERRED_DEVICE = {
            'device': device,
            'context': context,
            'queue': queue,
        }
