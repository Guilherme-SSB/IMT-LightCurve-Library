from setuptools import find_packages, setup

setup(
    name='imt_lightcurve',
    packages=find_packages(include=['imt_lightcurve']),
    version='1.0',
    description='Library created to manipulate LightCurves',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner', 'wheel'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
