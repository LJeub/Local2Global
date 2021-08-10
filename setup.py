from setuptools import setup, find_packages

setup(
    name='local2global',
    description='A Python implementation of the ASAP local2global algorithm.',
    url='https://github.com/LJeub/Local2Global.git',
    use_scm_version=True,
    setup_requires=['setuptools-scm'],
    author='Lucas G. S. Jeub',
    python_requires='>=3.5',
    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'pytest',
        'matplotlib',
        'networkx',
        'scikit-learn',
    ],
)
