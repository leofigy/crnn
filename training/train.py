#!/usr/bin/env python

# Not own by me self (done as part of CRNN tutorial and nestor's code)
import sys
import os

from datetime import datetime
from dataset import BacheDataset, BacheConfig
from mrcnn.model import MaskRCNN

def main():
    # params train_set
    if len(sys.argv) != 2:
        print("Provide the path for the dataset ...")
        return

    dataset_path = sys.argv[1]
    train_set, test_set  = BacheDataset(), BacheDataset()
    train_set.load_dataset(dataset_path)
    test_set.load_dataset(dataset_path, is_train=False)

    train_set.prepare()
    test_set.prepare()

    print("Train set %d" % len(train_set.image_ids))
    print("Test val set %d" % len(test_set.image_ids))

    # Configuration
    train_config = BacheConfig()

    train_config.display()

    # creating execution for current training
    timestamp = "output_at_%s" % datetime.now().strftime("%m-%d-%YT%H-%M-%S")
    self_path = os.path.dirname(os.path.realpath(__file__))
    weights = os.path.join(self_path, "artis", "mask_rcnn_coco.h5")
    
    outpath = os.path.join(self_path, timestamp)
    os.mkdir(outpath)


    # modeling creation
    model = MaskRCNN(mode='training', model_dir=outpath, config=train_config)
    model.load_weights(weights, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

    # start training
    model.train(train_set, test_set, learning_rate=train_config.LEARNING_RATE, epochs=5, layers='heads')

    return True

if __name__ ==  '__main__':
    if not main():
        sys.exit(1)