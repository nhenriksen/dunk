from setuptools import setup
from dunk.version import find_version

version_string = str(find_version())

setup(
    name='dunk',
    version=version_string,
    description='Tools for computing hydration free energies in AMBER.',
    url='https://github.com/nhenriksen/dunk',
    author='Niel M. Henriksen',
    author_email='shireham@gmail.com',
    license='MIT',
    packages=['dunk'],
    zip_safe=False,
    ### install_requires=['numpy', 'parmed??'],
)

