from Cython.Build import cythonize
from setuptools import Extension, setup


setup(
    name="binet_compiled",
    ext_modules=cythonize(
        [Extension("binet_compiled", ["binet.py"])],
        compiler_directives={"language_level": 3},
        build_dir="build/cython",
    ),
)
