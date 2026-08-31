# cgo-ae-src

## 1. Base image

```bash
docker pull nvcr.io/nvidia/pytorch:25.05-py3

docker run --gpus all -it --rm \
  --name cgo-ae-src \
  -v "$PWD/cgo-ae-src:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.05-py3 \
  bash
```

## 2. Miniconda (`/workspace/miniconda3`)

| Env | Python | Torch | Used for |
|-----|--------|-------|----------|
| `base` | 3.12 | 2.9.1+cu128 | Figures 5–7: our / torch / dynamo / tensorrt / tilelang / CUDA KV |
| `flashtensor` | 3.10 | 2.2.2 | Figure-7: flashtensor / tvm |

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p /workspace/miniconda3
rm /tmp/miniconda.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate base
```

## 3. Conda `base`

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate base

# tilelang first
pip install "tilelang==0.1.7"

# torch 2.9.1
pip install torch==2.9.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# triton LLVM
mkdir -p /root/.triton/llvm
cd /tmp
wget -O llvm-064f02da-ubuntu-x64.tar.gz \
  https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-064f02da-ubuntu-x64.tar.gz
tar -xzf llvm-064f02da-ubuntu-x64.tar.gz -C /root/.triton/llvm
printf '%s' \
  'https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-064f02da-ubuntu-x64.tar.gz' \
  > /root/.triton/llvm/llvm-064f02da-ubuntu-x64/version.txt

# triton + tilefusion
cd /workspace/triton
pip install -e . -v --no-build-isolation
cd /workspace/triton/3rd/fast-hadamard-transform
pip install -e . -v --no-build-isolation
cd /workspace/triton/tilefusion
pip install -e . -v --no-build-isolation

# mla_rope (figure-6)
cd /workspace/triton/tilefusion/ops/csrc/mla_rope
python setup.py build_ext --inplace
python setup_quant.py build_ext --inplace

# tensorrt
pip install "tensorrt-cu12==10.13.2.6" --extra-index-url https://pypi.nvidia.com
```

## 4. Conda `flashtensor`

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda create -n flashtensor python=3.10 -y
conda activate flashtensor

conda install -y -c conda-forge \
  llvm=18.1.2 mlir=18.1.2 llvmdev=18.1.2 libllvm18=18.1.2 cmake ninja
export LLVM_DIR="${CONDA_PREFIX}/lib/cmake/llvm"
export MLIR_DIR="${CONDA_PREFIX}/lib/cmake/mlir"

export PROJECT_DIR=/workspace/FlashTensor-AE
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"

pip install packaging==24.2 wheel==0.45.0 "torch==2.2.2"
bash ${PROJECT_DIR}/script/patch_pybind11.sh
bash ${PROJECT_DIR}/script/patch_onnxsim.sh
TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" pip install ${PROJECT_DIR} -v

cd ${PROJECT_DIR}/3rd/asuka
pip install -e . -v
pip uninstall -y triton triton-nightly || true
pip install -U --index-url \
  https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ \
  "triton-nightly==3.0.0.post20240708181524"

cd ${PROJECT_DIR}/3rd/tvm
mkdir -p build
cp ${PROJECT_DIR}/script/tvm_config.cmake build/config.cmake
cd build && cmake .. -G Ninja && ninja
pip install "xgboost==2.0.0"
cd ${PROJECT_DIR}/3rd/tvm/python
python setup.py install
```

## 5. Run

```bash
CUDA_VISIBLE_DEVICES=0 bash /workspace/ae/figure-5/run.sh
CUDA_VISIBLE_DEVICES=0 bash /workspace/ae/figure-6/run.sh
CUDA_VISIBLE_DEVICES=0 bash /workspace/ae/figure-7/run.sh
```
