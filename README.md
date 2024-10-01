[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UchechukwuUdenze/NHS_PostProcessing/HEAD)

==When using the binder link, navigate to docs, source and then notebooks==


## Introduction

This is a python Library to enable the National Hydrological Services carry out post processing operations quickly and efficiently by providing them with the exact features and packages that are required.


## Usage

### Prerequisites

As a first step you need a Python environment with all required dependencies. The recommended way is to use Anaconda and to create a new environment using our predefined environment files in [environments](https://github.com/Udenze-Uchechukwu/NHS_PostProcessing/tree/main/environments).

Use: 
```
conda env create -f environments/environment.yml
```

### Installation

**The Library is not yet avaialable on PyPi so it wll have to be installed directly from the git repo**

To install the library use:
```
pip install git+https://github.com/Udenze-Uchechukwu/NHS_PostProcessing.git
```

If you want to install an editable verson to implememt your own models or dataset you'll have to clone the repository  using:
```
git clone https://github.com/Udenze-Uchechukwu/NHS_PostProcessing.git
```

or just download the zip file [here](https://github.com/Udenze-Uchechukwu/NHS_PostProcessing/archive/refs/heads/main.zip)

After this, you are then left with a directory called *NHS_PostProcessing* or *NHS_PostProcessing-main*.  Next, we’ll go to that directory and install a local, editable copy of the package:
```
cd NHS_PostProcessing
pip install -e .
```

#### Note:

This is a work in progress and I am very open to suggestions and ideas
- Documentation: [NHS_postprocessing ReadTheDocs](https://nhs-postprocessing.readthedocs.io/en/latest/)
- Bug reports/Feature Requests: https://github.com/Udenze-Uchechukwu/NHS_PostProcessing/issues

## Contact

For questions, inquiries regarding the usasge of these repositories feel free to use the [discussion](https://github.com/Udenze-Uchechukwu/NHS_PostProcessing/discussions) board or reach me by email Uchechukwu.Udenze@ec.gc.ca if its urgents or requires special attention. For bug reports and feature requests, please open an [issue](https://github.com/Udenze-Uchechukwu/NHS_PostProcessing/issues) on the GitHub page
