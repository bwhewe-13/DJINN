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

import os

from setuptools import find_packages, setup


def read_readme():
    path = os.path.join(os.path.dirname(__file__), "README.md")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def load_pyproject_metadata():
    """Load [project] metadata from pyproject.toml.

    Tries to use stdlib tomllib (Python 3.11+), falls back to the toml
    package if available. Returns a dict of metadata or an empty dict
    on failure.
    """
    pyproject_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    if not os.path.exists(pyproject_path):
        return {}

    data = None
    try:
        # Python 3.11+
        import tomllib as _toml

        with open(pyproject_path, "rb") as f:
            data = _toml.load(f)
    except Exception:
        try:
            import toml as _toml

            with open(pyproject_path, "r", encoding="utf-8") as f:
                data = _toml.load(f)
        except Exception:
            data = None

    if not data:
        return {}

    # PEP 621: metadata is under the 'project' table
    return data.get("project", {})


metadata = load_pyproject_metadata()

setup_kwargs = {}

# Basic fields with sensible fallbacks to the original values
setup_kwargs["name"] = metadata.get("name", "djinnml")
setup_kwargs["version"] = metadata.get("version", "1.0")
setup_kwargs["description"] = metadata.get(
    "description", "Deep Jointly-Informed Neural Networks"
)
setup_kwargs["long_description"] = read_readme()
setup_kwargs["classifiers"] = metadata.get(
    "classifiers", ["Programming Language :: Python :: 3.7"]
)
setup_kwargs["keywords"] = " ".join(
    metadata.get("keywords", ["regression", "neural networks", "pytorch"])
)
setup_kwargs["url"] = (metadata.get("urls") or {}).get(
    "Homepage", "https://github.com/LLNL/djinn"
)

# Authors: take first author if present
authors = metadata.get("authors") or []
if authors and isinstance(authors, list):
    first = authors[0]
    setup_kwargs["author"] = first.get("name")
    setup_kwargs["author_email"] = first.get("email")
else:
    setup_kwargs["author"] = "Kelli Humbird"
    setup_kwargs["author_email"] = "humbird1@llnl.gov"

setup_kwargs["license"] = (
    metadata.get("license", {}).get("text", "LLNL")
    if isinstance(metadata.get("license"), dict)
    else metadata.get("license", "LLNL")
)

setup_kwargs["packages"] = find_packages()

# Dependencies: map to install_requires; metadata 'dependencies' is a list of strings
setup_kwargs["install_requires"] = metadata.get(
    "dependencies",
    [
        "torch>=1.8",
        "scipy",
        "scikit-learn>=1.5.1",
    ],
)

# Test extras
setup_kwargs["test_suite"] = "nose.collector"
setup_kwargs["tests_require"] = metadata.get("optional-dependencies", {}).get(
    "testing", ["nose", "nose-cover3"]
)

setup_kwargs.update(
    {
        "include_package_data": True,
        "zip_safe": False,
    }
)

setup(**setup_kwargs)
