#!/bin/bash

# nox internally uses:
#   virtualenv </some/temp/location> -p pythonX.Y
# virtualenv internally caches discovered pythons at 
#   $HOME/.local/share/virtualenv/py_info/1

# Due to virtualenv not playing well with pyenv to discover environments
# (see https://github.com/pypa/virtualenv/issues/1643)
# we have this workaround where we loop through available pyenvs and force
# virtualenv to create an environment based on it, thus forcing cache creation

eval "$(pyenv init -)"

python=$(pyenv which python)
for version in $(pyenv versions --bare)
do
    pyenv shell $version
    $python -m virtualenv -p $PYENV_VERSION /tmp/.env
done
