import os
import setuptools

description = 'Simulates the time evolution of a 1-dimensional wave packet in an arbitrary time independent potential'

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='wave_packet_dynamics',
    version='0.1.0',
    url='https://github.com/Rastow/wave-packet-dynamics',
    author='Robert Grzonka',
    author_email='robert.grzonka@fau.de',
    description=description,
    long_description=long_description,
    long_description_content_type='text/x-rst',
    keywords=["quantum-mechanics", "quantum-chemistry", "physics-simulation", "schroedinger-equation"],
    packages=setuptools.find_packages(),
    license='MIT',
    python_requires='>=3.8',
    install_requires=required,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
