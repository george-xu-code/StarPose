
import os
os.system('python tools/train.py configs/body_2d_keypoint/simcc/coco/starpose-18_1e-3_260e_256x192_simcc2.py --amp --cfg-options randomness.seed=3407')
