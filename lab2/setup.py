import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


root = Path(__file__).resolve().parent.parent
os.chdir(root)

setup(
    name="lab2-algorithms",
    packages=["lab1", "lab2"],
    ext_modules=cythonize(
        [
            Extension("lab1.binet_compiled", ["lab1/binet.py"]),
            Extension("lab1.strassen_compiled", ["lab1/strassen.py"]),
            Extension("lab2.laderman_compiled", ["lab2/laderman.py"]),
            Extension("lab2.multiplication_compiled", ["lab2/multiplication.py"]),
            Extension("lab2.inverse_compiled", ["lab2/inverse.py"]),
            Extension("lab2.lu_compiled", ["lab2/lu.py"]),
        ],
        compiler_directives={"language_level": 3},
        build_dir="build/cython",
    ),
)
