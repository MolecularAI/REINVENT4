REINVENT 4
==========


Description
-----------

REINVENT is a molecular design tool for de novo design, scaffold hopping,
R-group replacement, linker design, molecule optimization, and other small
molecule design tasks.  At its heart, REINVENT uses a Reinforcement Learning
(RL) algorithm to generate optimized molecules compliant with a user defined
property profile defined as a multi-component score. See AUTHORS.md for
paper references.


Requirements
------------

REINVENT is being developed on and for Linux but is also known to work on
MacOSX.  The code is written in Python 3 (>= 3.10).  The list of dependencies
can be found in the repository (see also Installation below). A GPU is not
strictly necessary but it is stronlgy recommended to run on GPU for
performance reasons.  Note that if no GPU is installed in your computer the
code attempts to run on the CPU only.


Installation
------------

1. Clone this Git repository.
2. Install compatible version of Python, for example with [Conda](https://conda.io/projects/conda/en/latest/index.html) (Docker, pyenv, or system package manager would work too).
    ```shell
    conda create --name reinvent4 python=3.10
    conda activate reinvent4
    ```
3. Go to the repository. Install the dependencies from the lockfile:
    ```shell
    pip install -r requirements-linux-64.lock
    ```
4. Install the tool. Dependencies were already installed in the previous step, no need to install them again (flag `--no-deps).  If you want to install in editable (development i.e changes to code are automatically picked up) mode add -e.
    ```shell
    pip install --no-deps . 
    ```
5. Use the tool. Installer added script `reinvent` to the path.
    ```shell
    reinvent --help
    ```

Updating dependencies
---------------------

Update lockfiles with [pip-tools](https://github.com/jazzband/pip-tools) (do not edit lockfiles manually):
```shell
pip-compile --extra-index-url=https://download.pytorch.org/whl/cu113 --extra-index-url=https://pypi.anaconda.org/OpenEye/simple --resolver=backtracking pyproject.toml
```
To update single package, use `pip-compile --upgrade-package somepackage` (see pip-tools docs).


Usage
-----

For the time being go through the files in config/toml where you will find
various examples on how to run REINVENT.  The files in config/json are
conversions from the JSON files and are functionally equivalent.

<!--- For concrete examples, you can check out the Jupyter notebook examples in the ReinventCommunity repo.
Running each example will result in a template file.There are templates for many running modes. 
Each running mode can be executed by `python input.py some\_running\_mode.json` after activating the environment.
    
Templates can be manually edited before using. The only thing that needs modification for a standard run are the file 
and folder paths. Most running modes produce logs that can be monitored by tensorboard. --->


Tutorials / `jupyter` notebooks
-------------------------------

NOTE: these will be updated at a later time!

<!--- There is another repository containing useful `jupyter` notebooks related to `REINVENT` 
called [ReinventCommunity](https://github.com/MolecularAI/ReinventCommunity). Note, that it uses a
different `conda` environment to execute, so you have to set up a separate environment. --->


Tests 
-----

The REINVENT project uses the `pytest` framework for its tests; before you run them you first have to create a 
configuration, which the tests will use.

In the project directory, create a `config.json` file in the `configs/` directory; you can use the example 
config (`example.config.json`) as a base.  Make sure that you set `MAIN\_TEST\_PATH` to a non-existent directory; it
is where temporary files will be written during the tests; if it is set to an existing directory, that directory 
will be removed once the tests have finished.

Some tests require a proprietary OpenEye license; you have to set up a few things to make the tests read your
license.  The simple way is to just set the `OE\_LICENSE` environment variable to the path of the file containing the
license.  

Once you have a configuration and your license can be read, you can run the tests.

```
$ pytest tests
```

Scoring Plugins
---------------

The scoring component of the code uses a simple plugin mechanism.  If you
wish to write your own, follow these instructions

1. Create */top/dir/somewhere/reinvent\_plugins/components* where */top/dir/somewhere* is a convenient location for you.
2. Do **not** place a \_\_init\_\_.py in either *reinvent\_plugins* or *components* as this would break the mechanism.  It is fine to create normal packages within *components* as long as you import those correctly.
3. Place a file whose name starts with *comp\_* into *reinvent\_plugins/components*.  The directory will be searched recursively so structure your code as needed.  Files with different names will be ignored.
4. Tag the scoring component class(es) in that file with the @add\_tag decorator.  More than one component class can be added to the same *comp\_* file. See existing code.
5. Tag at most one dataclass as parameter in the same file, see exisiting code.  This is optional.
6. There is no need to touch any of the REINVENT code.
7. Set or add */top/dir/somewhere* to PYTHONPATH or use any other mechanism to extend sys.path.
8. The scoring component should now be automatically picked up by REINVENT.
