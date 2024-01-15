# Runs whole preprocessing pipeline
import argparse
import GeoPatchify.snap as snap
import GeoPatchify.split as split
import preprocessing.remove_empty as remove_empty
import preprocessing.split_dataset as split_dataset
import preprocessing.crop as crop

def main(tif_folder, geojson_folder, patch_size, tile_size, seed):

    # Split tif files into non-overlapping tiles
    crop.main(tif_folder, "./temp/split_tiles/", tile_size)
    # Remove empty tiles
    remove_empty.main("./temp/split_tiles/", geojson_folder, "./temp/valid_tiles/")
    # Create train, eval, test splits
    split_dataset.main("./temp/valid_tiles/", seed, "./temp/split_train_eval_test/")
    
    # Create Masks with centered geometries
    for dataset_type in ["train", "eval", "test"]:
        masker = snap.Snap("./temp/split_train_eval_test/" + dataset_type + "/", geojson_folder, "./temp/snap/" + dataset_type + "/images", "./temp/snap/" + dataset_type + "/masks")
        masker.process()
    # Split masks and images into patches
    for dataset_type in ["train", "eval", "test"]:
        patchify = split.Split("./temp/snap/" + dataset_type + "/images", "./temp/snap/" + dataset_type + "/masks", "./temp/load/img/" + dataset_type, "./temp/load/masks/" + dataset_type, patch_size)
        patchify.process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tif-folder', help="Folder containing tif files")
    parser.add_argument("--geojson-folder", help="Folder containing geojson files")
    parser.add_argument("--patch-size", default=256, help="Size of patches (Final inputs for model)")
    parser.add_argument("--tile-size", default=2009, help="Size of tiles cropped from tif files before patching")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random train-eval-test set generator (before patching)")

    args = parser.parse_args()
    
    tif_folder = args.tif_folder 
    geojson_folder = args.geojson_folder
    patch_size = args.patch_size
    tile_size = args.tile_size
    seed = args.seed

    main(tif_folder, geojson_folder, patch_size, tile_size, seed)