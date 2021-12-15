#!/bin/bash
set -e
$SET_X

if [ x"$1" == "x" ] ; then
  echo "Usage: $0 install|dontinstall"
  exit 255
fi


/bin/rm -rf build dist mlflow_kernel.egg-info
python3 ./setup.py sdist bdist_wheel
/bin/rm -rf build mlflow_kernel.egg-info

if [ "$1" == "install" ] ; then
  pip uninstall -y mlflow_kernel
  pip install dist/mlflow_kernel-[0-9].[0-9].[0-9]*-py3-none-any.whl
fi

exit 0
