from tensorboard import program
import os

wd = f"{os.getcwd()}/runs"

# Set working directory to the latest subdirectory in runs
subdirs = [d for d in os.listdir(wd) if os.path.isdir(os.path.join(wd, d))]
latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(wd, d)))
wd = os.path.join(wd, latest_subdir)

print("Current working directory to visualise in tensorboard:", wd)

logdir = f"{wd}/tb"
tb = program.TensorBoard()
tb.configure(argv=[None, "--bind_all", "--logdir", logdir, "--port", "8089"])
url = tb.launch()

print(f"TensorBoard is running at {url}")
input("Press Enter to terminate...")
