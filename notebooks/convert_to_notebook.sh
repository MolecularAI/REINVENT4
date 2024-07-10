#!/bin/sh
#
# convert light-script Python files to Notebook
#


py_files='Reinvent_demo.py Reinvent_TLRL.py'

jupytext="$(which jupytext 2> /dev/null)"

if [ $? -ne 0 ]; then
    echo "Please install jupytext into your Python environment" 1>&2
    exit 1
fi    

for py_file in $py_files; do
    ipynb_file="$(echo $py_file | sed -e 's/\.py$/.ipynb/')"

    echo "Converting $py_file to $ipynb_file"
    jupytext --to ipynb -o "$ipynb_file" "$py_file"  
done
