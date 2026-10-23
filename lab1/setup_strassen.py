from Cython.Build import cythonize
from setuptools import Extension, setup


setup(
    name="strassen_compiled",
    ext_modules=cythonize(
        [Extension("strassen_compiled", ["strassen.py"])],
        compiler_directives={"language_level": 3},
        build_dir="build/cython",
    ),
)
