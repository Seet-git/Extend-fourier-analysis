# Extend_fourier_analysis

Install GPU torch for RTX 5070Ti

```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

[Windows version](https://www.youtube.com/watch?v=_Ua-d9OeUOg&pp=ygUOaW5zdGFsbCBsaW51eCDSBwkJ3goBhyohjO8%3D)

# Classification

```bash
  pip install -r requirements.txt
```

# Segmentation

### SAM2

SAM2 requires `python>=3.10`, `torch>=2.5.1`, and `torchvision>=0.20.1`.

```bash
# Clone official SAM2 repo
mkdir -p external
git clone https://github.com/facebookresearch/sam2.git external/sam2

# Install SAM2 from the official repo
cd external/sam2
pip install -e .
```

```bash
# Download checkpoints:
cd checkpoints
./download_ckpts.sh
cd ../../..
```

```bash
# If the SAM2 CUDA extension fails to build, install without it:
cd external/sam2
SAM2_BUILD_CUDA=0 pip install -e .
cd ../..
```

```bash
# Check installation: 
python -c "import torch; import sam2; print('CUDA:', torch.cuda.is_available())"
```

Run `train.sh` and `eval_fourier.sh`