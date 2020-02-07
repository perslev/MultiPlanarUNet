from setuptools import setup, find_packages

from mpunet import __version__

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open("requirements.txt") as req_file:
    requirements = list(filter(None, req_file.read().split("\n")))


setup(
    name='mpunet',
    version=__version__,
    description='Multi-Planar UNet for autonomous '
                'segmentation of 3D medical images',
    long_description=readme + "\n\n" + history,
    long_description_content_type='text/markdown',
    author='Mathias Perslev',
    author_email='map@di.ku.dk',
    url='https://github.com/perslev/MultiPlanarUNet',
    license='LICENSE.txt',
    packages=find_packages(),
    package_dir={'mpunet':
                 'mpunet'},
    include_package_data=True,
    setup_requires=["setuptools_git>=0.3",],
    entry_points={
       'console_scripts': [
           'mp=mpunet.bin.mp:entry_func',
       ],
    },
    install_requires=requirements,
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'Operating System :: POSIX',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'License :: OSI Approved :: MIT License']
)
