## SAM2-Adapter: Evaluating & Adapting Segment Anything 2 in Downstream Tasks: Camouflage, Shadow, Medical Image Segmentation, and More

## Environment
This code was implemented with Python 3.11 and PyTorch 2.3.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```


## Quick Start
1. Download the dataset and put it in ./load.
2. Download the pre-trained [SAM 2(Segment Anything)](https://github.com/facebookresearch/segment-anything-2) and put it in ./pretrained.
3. Training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 loadddptrain.py --config configs/demo.yaml
```
!Please note that the SAM model consume much memory. We use 4 x A100 graphics card for training. If you encounter the memory issue, please try to use graphics cards with larger memory!


4. Evaluation:
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run train.py --nnodes 1 --nproc_per_node 4 --config [CONFIG_PATH]
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Dataset

### Camouflaged Object Detection
- **[COD10K](https://github.com/DengPingFan/SINet/)**
- **[CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)**
- **[CHAMELEON](https://www.polsl.pl/rau6/datasets/)**

### Shadow Detection
- **[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)**

### Polyp Segmentation - Medical Applications
- **[Kvasir](https://datasets.simula.no/kvasir-seg/)**

## Cite

If you find our work useful in your research, please consider citing:

```

@inproceedings{chen2023sam,
  title={Sam-adapter: Adapting segment anything in underperformed scenes},
  author={Chen, Tianrun and Zhu, Lanyun and Deng, Chaotao and Cao, Runlong and Wang, Yan and Zhang, Shangzhan and Li, Zejian and Sun, Lingyun and Zang, Ying and Mao, Papa},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3367--3375},
  year={2023}
}

@misc{chen2024sam2adapterevaluatingadapting,
      title={SAM2-Adapter: Evaluating & Adapting Segment Anything 2 in Downstream Tasks: Camouflage, Shadow, Medical Image Segmentation, and More}, 
      author={Tianrun Chen and Ankang Lu and Lanyun Zhu and Chaotao Ding and Chunan Yu and Deyi Ji and Zejian Li and Lingyun Sun and Papa Mao and Ying Zang},
      year={2024},
      eprint={2408.04579},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.04579}, 
}


@misc{chen2023sam,
      title={SAM Fails to Segment Anything? -- SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, and More}, 
      author={Tianrun Chen and Lanyun Zhu and Chaotao Ding and Runlong Cao and Shangzhan Zhang and Yan Wang and Zejian Li and Lingyun Sun and Papa Mao and Ying Zang},
      year={2023},
      eprint={2304.09148},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}


```
