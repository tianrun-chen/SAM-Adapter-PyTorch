from masks import MaskMaker
from split_ma import Split
from np2png_ma import Np2Png
from rename_union_new_ma import RenameUnion
from proof_not_empty import Proof
from train_test_split_ma import TrainTestSplit
import create_folderstructure

create_folderstructure.create_folders('all')

maskmaker = MaskMaker('data/munich_geojson/PV_MUC_20220506_1_Florian.geojson','data/geodatenbayern_munich_lk_city','munich_florian')
maskmaker.process()
#bounding boxes not needed
#maskmaker.process_bounding_box()

split = Split(2500,512)
split.splitImages("data/masked_images_munich_florian")
#split.splitMask("data/munich_florian_bounding_masks")
split.splitMask("data/munich_florian_masks")

np2png_florian = Np2Png("data/split/munich_florian_masks","data/split/florian_png_masks")
np2png_florian.np_2_png()

rename_union = RenameUnion('data/split/florian_png_masks','data/split/rdy/masks','florian')
rename_union.rename_union()
rename_union = RenameUnion('data/split/masked_images_munich_florian','data/split/rdy/images','florian')
rename_union.rename_union()

maskmaker_tepe = MaskMaker('data/munich_geojson/20220509_PV_Labeling_Tepe.geojson','data/geodatenbayern_munich_lk_city','munich_tepe')
maskmaker_tepe.process()
#bounding boxes not needed
#maskmaker_tepe.process_bounding_box()

split_tepe = Split(2500,512)
split_tepe.splitImages("data/masked_images_munich_tepe")
#split_tepe.splitMask("data/munich_tepe_bounding_masks")
split_tepe.splitMask("data/munich_tepe_masks")

np2png_tepe = Np2Png("data/split/munich_tepe_masks", "data/split/tepe_png_masks")
np2png_tepe.np_2_png()

rename_union = RenameUnion('data/split/tepe_png_masks','data/split/rdy/masks','tepe')
rename_union.rename_union()
rename_union = RenameUnion('data/split/masked_images_munich_tepe','data/split/rdy/images','tepe')
rename_union.rename_union()

proof = Proof('data/split/rdy/images', 'data/split/rdy/masks')
proof.is_empty_del()

test_train_split = TrainTestSplit
test_train_split.split('data/split/rdy/images','load/img')

test_train_split.split('data/split/rdy/masks','load/masks')