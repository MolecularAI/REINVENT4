# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Maize workflow visualization

# %%
from maize.core.workflow import Workflow

# ugly start imports ... but do the trick
from maize.steps.mai import *
from maize.graphs.mai import *

# %%
wf = Workflow.from_file("autodock_gpu.yaml")
g = wf.visualize()
g

# %%
g.save("autodock_gpu.dot")
