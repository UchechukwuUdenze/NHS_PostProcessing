from pathlib import Path
from setuptools import find_packages, setup

readme_file = Path(__file__).absolute().parent / "README.md"
with readme_file.open("r") as fp:
    long_description = fp.read()

version = {}
with open("postprocessinglib/__version__.py", "r") as fp:
    exec(fp.read(), version)

setup(
    name='nhs_posprocessing',
    version=version["__version__"],
    packages=[
            'postprocessinglib', 
            'postprocessinglib.evaluation',
            'postprocessinglib.utilities'
    ],
    url='https://nhs-postprocessing.readthedocs.io/en/stable/', 
    project_urls={
        'Visualization': 'https://github.com/users/UchechukwuUdenze/projects/4',
        'Data Manipulation': 'https://github.com/users/UchechukwuUdenze/projects/2',
        'Documentation': 'https://github.com/users/UchechukwuUdenze/projects/5',
    },  
    author='Uchechukwu Udenze',
    author_email='uchechukwu.udenze@ec.gc.ca',
    description='NHS Post Processing library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'netcdf4',
        'matplotlib',
        'geopandas',
        'shapely',
        'openpyxl'
    ],
    test_suite='tests',
    license='MIT'
)
