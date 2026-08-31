set -x 
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "  : $PROJECT_DIR"

git submodule init
git submodule update

pip install tilelang
cd $PROJECT_DIR/3rd/fast-hadamard-transform
pip install . -v 
cd $PROJECT_DIR
pip install -e . -v 
cd $PROJECT_DIR/tilefusion
pip install -e . -v 