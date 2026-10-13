from Cython.Build import cythonize
from setuptools import Extension, setup


setup(
    name="alphatensor_compiled",
    ext_modules=cythonize(
        [
            Extension("strassen_compiled", ["strassen.py"]),
            Extension("alphatensor_compiled", ["alphatensor.py"]),
        ],
        compiler_directives={"language_level": 3},
        build_dir="build/cython",
    ),
)
