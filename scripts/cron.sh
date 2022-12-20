#!/bin/bash

# Invoked regularly in a crontab
# with parameters (1) Conda env name, (2) Streamlit base directory
# ---------- 
# crontab -e
# ---------- 
# SHELL=/bin/bash
# BASH_ENV=/path/to/file/with/contents/of/conda/init
# 0 * * * * bash /path/to/this/script <dev_conda_env> /path/to/dev/streamlit
# 0 * * * 7 bash /path/to/this/script <prod_conda_env> /path/to/prod/streamlit
# ----------

CONDA_ENV=$1
STREAMLIT_DIR=$2
STREAMLIT_CMD="python -m streamlit run app/main.py" 

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}
cd ${STREAMLIT_DIR}
echo "${CONDA_PREFIX}"
git pull
# No need to reinstall the package since its initially installed
# in development mode anyway. This means we won't be up-to-date
# with the dependencies, so we'll handle that offline as and when
# the dependencies change.
# pip install -e .

# The following commented-out block is the rather drastic (formerly used)
# step of killing the streamlit process and starting it again.
# Besides being disruptive, recent versions of streamlit seem to monitor
# changes in the 'magnet' package and reflect them immediately, so none of this
# is needed.
: '
echo "Checking for running streamlit processes with CWD=${STREAMLIT_DIR}"

PIDS=`ps -ef | grep "${STREAMLIT_CMD}" | grep -v grep | tr -s ' ' | cut -d ' ' -f 2`
for PID in $PIDS; do
    PID_WD=`pwdx ${PID} | cut -d ' ' -f 2`
    if [ "$PID_WD" = "$STREAMLIT_DIR" ]; then
      echo "Found PID ${PID}. Trying to stop .."
      kill ${PID}
    fi
done

echo "Re-running command .."
eval "${STREAMLIT_CMD}" &>/dev/null & disown;
'

conda deactivate

# Start PLECS if not running 

# "PLECS_server -v" returns with an exit code of 0 whether its running or not
# so we check its stdout instead.
if /opt/plecs/PLECS_server -v | grep -q 'not running'; then
  /opt/plecs/PLECS_server
fi
