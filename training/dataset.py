import os
from mrcnn.utils import Dataset

class BacheDataset(Dataset):
    # class items just in case you want to change it
    root_name = "dataset"
    class_name = "baches"

    # format for folders is
    # /images , /annots
    images_dir = "images"
    annots_dir = "annots"
    proportion = 20 # 80 - 20 (train - test)

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class(BacheDataset.root_name, 1, BacheDataset.class_name)
        images_dir = os.path.join(dataset_dir, BacheDataset.images_dir)
        annots_dir = os.path.join(dataset_dir, BacheDataset.annots_dir)

        # getting all files
        items = [filename for filename in os.listdir(images_dir)]

        # dividing set
        test_size = ( len(items) / 100 ) * 20
        train_size = int(len(items) - test_size)
        target_set = items[:train_size] if is_train else items[train_size:]

        def addX(filename):
            name, ext = os.path.splitext(filename)
            image_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annots_dir, name + ".xml")
            self.add_image(BacheDataset.root_name, image_id=name, path=image_path, annotation=ann_path)

        for image in target_set:
            addX(image)

    # xml reader
    def extract_boxes(self, filename):
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # loader
    def load_mask(self, image_id):
        info = self.images_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(BacheDataset.class_name))
        return masks, asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
