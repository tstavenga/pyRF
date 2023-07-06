from setuptools import setup, find_packages

setup(
    name='pyRF',
    version='0.0',
    license='',
    author='Thijs Stavenga',
    author_email='thijsstavenga@msn.com',
    description=(
        "Package for simulating the eigenmodes of RF circuits and resonators. Install using the pip editable package command: python -m pip install -e ."
    ),
    packages=find_packages(),
    classifiers=['Development Status :: 2 - Pre-Alpha', 'Intended Audience :: Science/Research', 'Programming Language :: Python :: 3.9'],
    keywords='RF qubit resonator hfss eigenmode script scripting physics',
    install_requires=['dictdiffer', 'pydrive2'],
    url='https://github.com/tstavenga/pyRF'
)

