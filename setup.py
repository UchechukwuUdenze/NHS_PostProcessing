from pathlib import Path
from setuptools import find_packages, setup

readme_file = Path(__file__).absolute().parent / "README.md"
with readme_file.open("r") as fp:
    long_description = fp.read()

version = {}
with open("postprocessinglib/__version__.py", "r") as fp:
    exec(fp.read(), version)

setup(
    name='postprocessinglib',
    version=version["__version__"],
    packages=find_packages(include=[
            'postprocessinglib.evaluation',
            'postprocessinglib.utilities'
    ]), 
    url='https://postprocessinglib.readthedocs.io', 
    # project_urls={
    #     'Source': 'https://github.com/repo/repo/repo',
    # },  
    author='Uchechukwu Udenze',
    description='post processing library',
    long_description=long_description,
    python_requires='>=3.11',
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'netcdf4'
    ],
    test_suite='tests',
)