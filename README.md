# Environment:
```bash
pip install -r requirements.txt
mim install mmcv
```


# Training:
```python
torchrun train.py --config configs/sam-vit-b.yaml
```


# Resampling Modes
```python
NEAREST = "nearest"
BILINEAR = "bilinear"
BICUBIC = "bicubic"
# For PIL compatibility
BOX = "box"
HAMMING = "hamming"
LANCZOS = "lanczos"
```
    
