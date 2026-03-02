###############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).
#
# LLNL-CODE-754815
#
# All rights reserved.
#
# This file is part of DJINN.
#
# For details, see github.com/LLNL/djinn.
#
# For details about use and distribution, please read DJINN/LICENSE.
###############################################################################

"""Build entry point for setuptools.

Project metadata and configuration live in ``pyproject.toml``; this file is a
minimal compatibility shim for tools that still invoke ``setup.py`` directly.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
