#!/usr/bin/env python

# Authored by Nestor Salvador Martinez

import sys
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.model import MaskRCNN
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from dataset import BacheDataset, PredictionConfig
from mrcnn import visualize

def main():
    # function will look for images
    if len(sys.argv) < 3:
        print("Please tell me the model path at list and the images ...")
        return

    precomputed = sys.argv[1]
    images = sys.argv[2:]
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model.load_weights(precomputed, by_name=True)
    
    # testing images
    for img in images:
        image = load_img(img)
        image = img_to_array(image)
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'baches'], r['scores'],
                            title="Predictions")


if __name__ == '__main__':
    if not main():
        sys.exit(1)