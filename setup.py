from setuptools import setup, find_packages

setup(
    name='higashi2cellscope',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'h5py',
        'numpy',
        'pandas',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'higashi_gen=higashi2cellscope.cli:main',
        ],
    },
)