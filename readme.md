# Important:
when using cpu/cuda for training, line 98 in sam.py has to be changed between 'cpu' and 'cuda'

# Short:
Requirements:
- ```console
    pip install -r requirements.txt
    mim install mmcv
    ```
- install [pytorch](https://pytorch.org/get-started/locally)

Preprocessing:
- Needed data:
- Folder with all *.geotif tiles of the respected area
- Folder with all *.geojson files containing polygons as masks, masking pv-panels in the area
- to Run:
    - ```console
        python run.py --runtype all --geotif path/to/geotiffolder/ --geojson path/to/geojsonfolder/ --splitsize 512
        ```
    - ```console
        python run.py --runtype split --geojson path/to/geojsonfolder --splitimages path/to/imagesfolder/ --splitmasks path/to/masksfolder/ --splitsize 512
        ```

Training:
- CPU/Cuda:
    - CPU: train_min_ma_checkpoint.py
    - Cuda: train_cuda.py
    - **! sam.py line 98 needs to be adjusted !**

- Needed data:
    - pretrained sam-weights
    - preprocessed test/eval images/masks (-->Preprocessing)
        - config File
- to Run:
    - ```console
        python train_min_ma_checkpoint.py --config configs/ma_B.yaml
        ```
    - ```console
        python -m torch.distributed.launch train_cuda.py --config configs/ma_B_cuda.yaml
        ```

Testing:
- Needed data:
    - trained weights (--> Training)
    - test images/masks
    - config File (equal to Training)
- to Run:
    - ```console
        python test_min.py --config configs/ma_B.yaml --model save/... 
        ```
    - ```console
        python test_cuda.py --config configs/ma_B_cuda.yaml --model save/... 
        ```
Postprocessing:
- Needed data:
    - predicted masks (--> Testing)
- to Run:
    - ma_show_results.py (Build overlay showing true/false positive, true/false negative + iou)
    - ma_show_loss.py (shows loss during training based on logs)
    - ma_make_overlay.py (Build img/mask overlay)
    - ma_make_hist.py (Build mask histogram)
    - ma_make_binary_50.py (Build Binary-Mask with 50% threshold)
    - ma_make_contrast.py (0-1 Pixelvalue Image --> 0-255 Pixelvalue Grayscale Image)

Forwardpass:
- Needed data:
    - trained weights (--> Training)
    - config File (equal to Training)
- to Run:
    ```console
    python forwardpass/run_forwardpass.py --lat_1 --lon_1 --lat_2 --lon_2 --config --model
    ```

# Detailed:
Preprocessing:
- masks.py: small changes to the code of Yasmin mainly taken from [fixMatchSeg-Muc](https://github.com/yasminhossam/fixMatchSeg-Muc/blob/main/solarnet/preprocessing/masks.py) plus added bounding box creation (not used)
- load_munich.py: processing the geotif and geojson files to get the needed data for the dicts Jasmin uses in her code to calculate the masks
        - readTifKoos(): reading the UTM-coordinates of the geotif (Tiles)
        - readGeoJsonPoly(): reading the mask (Polygon) coordinates
        - buildReadData(): finding the tiles which include the masks, returning the dict mask.py creates the np-masks from
        - copyTif(): copys the tifs which include a mask to a seperate Folder
- split_ma.py: calculates the split and splits images/masks the same way.
    - calcSplit(): calculates the pixel-coordinates where the masks/images have to be cutted to get the needed image size. The function should return coordinates which create some overlapping. The algorithm is pretty basic and there is no proof for "the best" split.
- np2png_ma.py: converting np-masks to png-binary images.
- rename_union_new_ma.py: includes a info in the mask/image name to unite them into one trainings-set.
- proof_not_empty.py: checks if there is no (or below a absolute threshold (20)) mask (binary ones) in a mask and if so delets image + mask
- train_test_split_ma.py: splits the masks/images randomly in test/train/eval folders.

Training:

Testing:

Postprocessing:


# Results:
- after 20 epochs training: [Dropbox](https://www.dropbox.com/scl/fo/fkaq4v9izj69md45fa6b6/h?rlkey=0dmuoq15f9n3s2fohkvt1etz6&dl=0) 
(DOP: Bayerische Vermessungsverwaltung – [www.geodaten.bayern.de](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dop40) (Daten verändert), Lizenz: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.de))
    - mean IoU over 88 test-images: 0.47234

- after 30 epochs training: [Dropbox](https://www.dropbox.com/scl/fi/i84e584qwqv15oo2ohspq/32682_5340_54_tepe.png?rlkey=jdlfuqil8d3rk7f9ylg0ypvh8&dl=0)
(DOP: Bayerische Vermessungsverwaltung – [www.geodaten.bayern.de](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dop40) (Daten verändert), Lizenz: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.de)) 
    - mean IoU over 88 test-images: 0.47306

# Information:
- pyTorch-SAM-adapter based on [Sam-Adapter-Pytorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)
- supported by [Fortiss](https://fortiss.org)