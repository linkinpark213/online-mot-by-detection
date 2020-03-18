import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def init_coco_dict():
    return {
        'info': {
            'description': 'MOTSChallenge instance segmentation dataset',
            'url': 'https://www.vision.rwth-aachen.de/page/mots',
            'version': '1.0',
            'year': 2020,
            'contributor': 'linkinpark213',
            'date_created': '2020/3/5'
        },
        'licenses': [
            {
                'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                'id': 1,
                'name': 'Attribution-NonCommercial-ShareAlike License'
            }
        ],
        'images': [],
        'annotations': [],
        'categories': [
            {
                'supercategory': 'person',
                'id': 1,
                'name': 'person'
            }
        ]
    }


def mots_to_coco(mot_path):
    train_dict = init_coco_dict()
    val_dict = init_coco_dict()

    image_count = 0
    instance_count = 0

    images_path = os.path.join(mot_path, 'images')
    instances_path = os.path.join(mot_path, 'instances')

    seqs = os.listdir(images_path)
    seqs.sort()

    for seq in seqs:
        print('Processing sequence {}'.format(seq))
        image_filenames = os.listdir(os.path.join(images_path, seq))
        image_filenames.sort()

        for image_ind, image_filename in enumerate(image_filenames):
            if (image_filename[-3:] != 'jpg'):
                print('Ignoring file {}'.format(image_filename))
                continue
            print('Processing image {}/{}'.format(seq, image_filename))
            image_count += 1
            instance_filename = image_filename[:-3] + 'png'
            instance_img = np.array(Image.open(open(os.path.join(instances_path, seq, instance_filename), 'rb')))

            image_dict = {
                'license': 1,
                'file_name': seq + '/' + image_filename,
                'coco_url': '',
                'height': instance_img.shape[0],
                'width': instance_img.shape[1],
                'date_captured': '',
                'flickr_url': '',
                'id': image_count
            }

            if image_ind >= 0.8 * len(image_filenames):
                val_dict['images'].append(image_dict)
            else:
                train_dict['images'].append(image_dict)

            obj_ids = np.unique(instance_img)

            image = cv2.imread(os.path.join(images_path, seq, image_filename))
            for obj_id in obj_ids:
                if obj_id // 1000 == 2:
                    instance_count += 1
                    obj_instance_id = obj_id % 1000
                    instance_dict = {
                        'iscrowd': 0,
                        'image_id': image_count,
                        'category_id': 1,
                        'id': instance_count
                    }
                    mask = np.where(instance_img == obj_id, 1, 0).astype(np.uint8)
                    instance_dict['area'] = cv2.countNonZero(mask)
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

                    instance_dict['segmentation'] = [c.reshape(-1).tolist() for c in contours if len(c) > 2]
                    x1 = int(min([np.min(c[:, 0, 0]) for c in contours]))
                    y1 = int(min([np.min(c[:, 0, 1]) for c in contours]))
                    x2 = int(max([np.max(c[:, 0, 0]) for c in contours]))
                    y2 = int(max([np.max(c[:, 0, 1]) for c in contours]))
                    instance_dict['bbox'] = [x1, y1, x2, y2]
                    if len(instance_dict['segmentation']) > 0:
                        if image_ind >= 0.8 * len(image_filenames):
                            val_dict['annotations'].append(instance_dict)
                        else:
                            train_dict['annotations'].append(instance_dict)

            # image = image / 255.
            # plt.imshow(image)
            # plt.show()
            # break
    json.dump(train_dict, open('instances_train.json', 'w+'))
    json.dump(val_dict, open('instances_val.json', 'w+'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mots_path', type=str)
    args = parser.parse_args()
    mots_to_coco(args.mots_path)
