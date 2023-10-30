#!/bin/sh
#
# Create symlinks to support old model files
# Run in top level
#

ln -s reinvent/models reinvent_models

cd reinvent/models

ln -s reinvent reinvent_core
ln -s libinvent lib_invent
ln -s linkinvent link_invent
ln -s mol2mol molformer
