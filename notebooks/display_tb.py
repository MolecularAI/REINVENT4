# Start TensorBoard
from tensorboard import program
import os
# %load_ext tensorboard

wd = f"{os.getcwd()}/tmp"
logdir = f"{wd}/tb_stage1_0"
tb = program.TensorBoard()
tb.configure(argv=[None, "--bind_all", "--logdir", logdir, "--port", "8089"])
url = tb.launch()
print(wd)

print(f"TensorBoard is running at {url}")
input("Press Enter to terminate...")
