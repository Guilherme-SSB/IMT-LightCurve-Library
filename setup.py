from setuptools import setup

setup(
    name='imt_lightcurve',
    python_requires='>=3.7.11',
    packages=['imt_lightcurve', 'imt_lightcurve.models', 'imt_lightcurve.help_functions', 'imt_lightcurve.simulation', 'imt_lightcurve.visualization', 'imt_lightcurve.data_helper'],
    version='2.1',
    description='Library created to manipulate LightCurves',
    author='Guilherme Samuel',
    author_email='gui.samuel10@gmail.com',
    license='MIT',
    install_requires=['lightkurve==2.0.10', 'control==0.9.0'],
    setup_requires=['pytest-runner', 'wheel'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
