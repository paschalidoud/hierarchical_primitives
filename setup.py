"""Setup hierarchical primitives."""

from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from itertools import dropwhile
import numpy as np
from os import path


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("hierarchical_primitives", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def get_extensions():
    return cythonize([
        Extension(
            "hierarchical_primitives.fast_sampler._sampler",
            [
                "hierarchical_primitives/fast_sampler/_sampler.pyx",
                "hierarchical_primitives/fast_sampler/sampling.cpp"
            ],
            language="c++11",
            libraries=["stdc++"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-std=c++11", "-O3"]
        ),
        Extension(
            "hierarchical_primitives.external.libmesh.triangle_hash",
            sources=["hierarchical_primitives/external/libmesh/triangle_hash.pyx"],
            include_dirs=[np.get_include()],
            libraries=["m"]  # Unix-like specific
        )
    ])


def get_install_requirements():
    return [
        "numpy",
        "trimesh",
        "torch",
        "torchvision",
        "cython",
        "Pillow",
        "pyquaternion",
        "pykdtree",
        "matplotlib",
        "simple-3dviz"
    ]


def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="hierarchical_primitives",
        version=meta["version"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
        ],
        install_requires=get_install_requirements(),
        ext_modules=get_extensions()
    )


if __name__ == "__main__":
    setup_package()
