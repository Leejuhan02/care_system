PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src
echo "startstart"
echo GPIO_CHIP_NAME=0
sudo -E $PROJECT_ROOT/bin/python $PROJECT_ROOT/main.py
