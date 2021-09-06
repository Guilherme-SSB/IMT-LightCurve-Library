from setuptools import find_packages, setup

setup(
    name='IMT-LightCurve',
    packages=find_packages(include=['imt_lightcurve']),
    version='0.1.0',
    description='Library created to manipulate LightCurves',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)