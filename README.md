REINVENT 4
==========


Description
-----------

REINVENT is a molecular design tool for de novo design, scaffold hopping,
R-group replacement, linker design, molecule optimization, and other small
molecule design tasks.  At its heart, REINVENT uses a Reinforcement Learning
(RL) algorithm to generate optimized molecules compliant with a user defined
property profile defined as a multi-component score.  Transfer Learning (TL)
can be used to create or pre-train a model that generates molecules closer
to a set of input molecules. 

A paper describing the software has been published as Open Access in the
Journal of Cheminformatics:
[Reinvent 4: Modern AI–driven generative molecule design](https://link.springer.com/article/10.1186/s13321-024-00812-5?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20240221&utm_content=10.1186/s13321-024-00812-5).
See AUTHORS.md for references to previous papers.


Requirements
------------

REINVENT is being developed on and primarily for Linux and supports both GPU
and CPU.  The Linux version is fully validated.  REINVENT runs on Windows with
both GPU and CPU but this platform is mostly untested.  MacOSX is only
supported on the CPU.

The code is written in Python 3 (>= 3.10).  The list of
dependencies can be found in the repository (see also Installation below).

A GPU is not strictly necessary but strongly recommended for performance
reasons especially for transfer learning/model training.  It should be noted
that reinforcement learning (RL) requires the computation of scores.  Most scoring
components run on the CPU thus a GPU is of less importance for RL depending
on how much time is spent on the CPU.

Note that if no GPU is installed in your computer the code will run on the
CPU automatically.  REINVENT [supports](https://pytorch.org/get-started/locally/) NVIDIA and also some AMD GPUs.
For most design tasks a memory of about 8 GiB for both CPU main memory and
GPU memory is sufficient.


Installation
------------

1. Clone this Git repository.
2. Install a compatible version of Python, for example with [Conda](https://conda.io/projects/conda/en/latest/index.html) (other virtual environments like Docker, pyenv, or the system package manager would work too).
    ```shell
    conda create --name reinvent4 python=3.10
    conda activate reinvent4
    ```
3. Change directory into the repository and install the dependencies from the lockfile:
    ```shell
    pip install -r requirements-linux-64.lock
    ```
4. Optional: if you want to use **AMD GPUs** on Linux you would need to install the [ROCm PyTorch version](https://pytorch.org/get-started/locally/) manually _after_ installation of the dependencies in point 3, e.g.
   ```shell
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
   ```
5. Install the tool. The dependencies were already installed in the previous step, so there is no need to install them again (flag `--no-deps).  If you want to install in editable mode (changes to the code are automatically picked up) add -e before the dot.
    ```shell
    pip install --no-deps . 
    ```
6. Test the tool. The installer has added a script `reinvent` to your PATH.
    ```shell
    reinvent --help
    ```

Basic Usage
-----------

REINVENT is a command line tool and works principally as follows
```shell
reinvent -l sampling.log sampling.toml
```

This writes logging information to the file `sampling.log`.  If you wish to write
this to the screen, leave out the `-l sampling.log` part. `sampling.toml` is the
configuration file.  The main user format is [TOML](https://toml.io/en/) as it tends to be more
use friendly.  JSON can be used too, add `-f json`, but a specialised editor is
recommended as the format is very sensitive to minor changes.

Sample configuration files for all run modes are
located in `config/toml` of the repository and file paths therein would need to be
adjusted to your local installation.  In particular, ready made prior models are
located in `priors` and you would choose a model and the
appropriate run mode depending on the research problem you are trying to address.
There is additional information in `config/toml` in several `*.md` files with
instructions on how to configure the TOML file.


Tutorials / `Jupyter` notebooks
-------------------------------

Basic instructions can be found in the comments in the config examples in `config/toml`.

Notebooks will be provided in the `notebook/` directory.  Please note that we provide the notebooks in jupytext "light script" format.  To work with the light scripts you will need to install jupytext.  A few other packages will come in handy too.

```shell
pip install jupytext mols2grid seaborn
```

The Python files in `notebook/` can then be converted to a notebook e.g.

```shell
jupytext --to ipynb -o Reinvent_demo.ipynb Reinvent_demo.py
```


Updating dependencies
---------------------

Update the lock files with [pip-tools](https://pypi.org/project/pip-tools/) (please, do not edit the files manually):
```shell
pip-compile --extra-index-url=https://download.pytorch.org/whl/cu121 --extra-index-url=https://pypi.anaconda.org/OpenEye/simple --resolver=backtracking pyproject.toml
```
To update a single package, use `pip-compile --upgrade-package somepackage`
(see the documentation for pip-tools).


Scoring Plugins
---------------

The scoring subsystem uses a simple plugin mechanism (Python
[native namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#native-namespace-packages)).  If you
wish to write your own plugin, follow the instructions below.  The public
repository contains a [contrib](https://github.com/MolecularAI/REINVENT4/tree/main/contrib/reinvent_plugins/components) directory with some useful examples.

1. Create `/top/dir/somewhere/reinvent\_plugins/components` where `/top/dir/somewhere` is a convenient location for you.
2. Do **not** place a `__init__.py` in either `reinvent_plugins` or `components` as this would break the mechanism.  It is fine to create normal packages within `components` as long as you import those correctly.
3. Place a file whose name starts with `comp_*` into `reinvent_plugins/components`.   Files with different names will be ignored i.e. not imported. The directory will be searched recursively so structure your code as needed but directory/package names must be unique.
4. Tag the scoring component class(es) in that file with the @add\_tag decorator.  More than one component class can be added to the same *comp\_* file. See existing code.
5. Tag at most one dataclass as parameter in the same file, see existing code.  This is optional.
6. There is no need to touch any of the REINVENT code.
7. Set or add `/top/dir/somewhere` to the `PYTHONPATH` environment variable or use any other mechanism to extend `sys.path`.
8. The scoring component should now be automatically picked up by REINVENT.


Unit and Integration Tests 
--------------------------

This is primarily for developers and admins/users who wish to ensure that the
installation principally works.  The information here is not relevant to the
practical use of REINVENT.  Please refer to _Basic Usage_ for instructions on
how to use the `reinvent` command.

The REINVENT project uses the `pytest` framework for its tests.  Before you run
them you first have to create a configuration file which the tests will use.

In the project directory, create a `config.json` file in the `configs/` directory.
You can use the example config `example.config.json` as a base.  Make sure that
you set `MAIN_TEST_PATH` to a non-existent directory.  That is where temporary
files will be written during the tests.  If it is set to an existing directory,
that directory will be removed once the tests have finished.

Some tests require a proprietary OpenEye license.  You have to set up a few
things to make the tests read your license.  The simple way is to just set the
`OE_LICENSE` environment variable to the path of the file containing the
license.  

Once you have a configuration and your license can be read, you can run the tests.

```
$ pytest tests
```
