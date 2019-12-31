#!/usr/bin/env python

# Not own by me self (done as part of CRNN tutorial and nestor's code)
import sys
from dataset import BacheDataset
from mrcnn.config import Config

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
    train_config = Config()
    train_config.NAME = "bache_cfg"
    train_config.NUM_CLASSES = 2
    train_config.STEPS_PER_EPOCH = 131

    train_config.display()

    return True

if __name__ ==  '__main__':
    if not main():
        sys.exit(1)