import os
import time
import collections
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torch
import torchvision.transforms.functional as TF

from transform import box2hm
from utils import get_normalizer

CLASS2INDEX = {'hat': 0, 'person': 1, 'dog': 0}
INDEX2CLASS = {0: 'hat', 1: 'person'}
CLASS2COLOR = {0: (255, 0, 0), 1: (0, 255, 0)}

# reference: https://pytorch.org/docs/stable/_modules/torchvision/datasets/voc.html#VOCDetection
class VOC:
    def __init__(self, root, transform, image_set, pretrained, normalized_coord, num_cls):
        self.transform  = transform
        self.image_set  = image_set
        self.normalize  = get_normalizer(pretrained=pretrained)
        self.normalized_coord = normalized_coord
        self.num_cls    = num_cls
        
        image_dir       = os.path.join(root, 'JPEGImages')
        annotation_dir  = os.path.join(root, 'Annotations')
        splits_dir      = os.path.join(root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), 'r') as f:
            file_names = [x.strip() for x in f.readlines()]
            
        self.images      = [os.path.join(image_dir, x + '.jpg') for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + '.xml') for x in file_names]

        assert len(self.images) == len(self.annotations)
        print('%s: %d images are loaded from %s'%(time.ctime(), len(self.images), root))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_pil         = Image.open(self.images[index]).convert('RGB')
        voc_dict        = self._parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        box_lst, id_lst = self._parse_voc_dict(voc_dict)
                
        img_np, bbs_iaa = self._type_cast(img_pil, box_lst, id_lst)
        return img_np, bbs_iaa, voc_dict

    def _parse_voc_dict(self, voc_dict):
        box_lst, id_lst = [], []
        for obj in voc_dict['annotation']['object']:
            id_lst.append(CLASS2INDEX[obj['name'].lower()])

            bndbox  = obj['bndbox']
            box     = bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']
            box_lst.append(list(map(int, box)))
        return box_lst, id_lst

    def _parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self._parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    def _type_cast(self, img_pil, box_lst, id_lst):
        # change type of data to use `imgaug`
        img_np = np.asarray(img_pil)        
        
        bbs = []
        for (x1, y1, x2, y2), label in zip(box_lst, id_lst):
            bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
        bbs_iaa = BoundingBoxesOnImage(bbs, shape=img_np.shape)
                
        return img_np, bbs_iaa
    
    def collate_fn(self, batch):
        img_np_lst, bbs_iaa_lst, voc_dict_lst = list(zip(*batch))

        # batch-wise image augmentation
        img_np_lst, bbs_iaa_lst = self.transform(img_np_lst, bbs_iaa_lst)
        
        # type casting for heatmap
        batch_bbs_lst, batch_id_lst = [], []
        for bbs_iaa in bbs_iaa_lst:
            temp_bbs_lst, temp_id_lst = [], [] 
            for bbs in bbs_iaa.bounding_boxes:
                temp_bbs_lst.append(bbs.coords.flatten())
                temp_id_lst.append(bbs.label)
            batch_bbs_lst.append(temp_bbs_lst)
            batch_id_lst.append(temp_id_lst)

        # make heatmap
        heatmap_lst, offset_lst, wh_lst, mask_lst = [],[],[],[]
        for bbs_lst, id_lst in zip(batch_bbs_lst, batch_id_lst):
            heatmap, offset, wh, mask = box2hm(bbs_lst, id_lst, bbs_iaa_lst[0].shape[:2], num_cls=self.num_cls, normalized=self.normalized_coord)
            heatmap_lst.append(heatmap)
            offset_lst.append(offset)
            wh_lst.append(wh)
            mask_lst.append(mask)

        # to tensor
        batch_img_ten     = torch.stack([self.normalize(TF.to_tensor(img_np)) for img_np in img_np_lst])
        batch_heatmap_ten = torch.stack([torch.Tensor(heatmap) for heatmap in heatmap_lst])
        batch_offset_ten  = torch.stack([torch.Tensor(offset) for offset in offset_lst])
        batch_wh_ten      = torch.stack([torch.Tensor(wh) for wh in wh_lst])
        batch_mask_ten    = torch.stack([torch.Tensor(mask) for mask in mask_lst])
        
        return batch_img_ten, batch_heatmap_ten, batch_offset_ten, batch_wh_ten, batch_mask_ten, voc_dict_lst

class TrainAugmentor:
    def __init__(self, crop_percent=(0.0, 0.1), color_multiply=(1.2, 1.5), translate_percent=0.1,
                 affine_scale=(0.5, 1.5), multiscale_flag=False, multiscale=[320, 608, 32]):        
        
        self.multiscale_flag = multiscale_flag
        self.multiscale_min  = multiscale[0]
        self.multiscale_max  = multiscale[1]
        self.multiscale_step = multiscale[2]
                
        self.seq = iaa.Sequential([
                        iaa.Multiply(color_multiply),
                        iaa.Affine( 
                            translate_percent=translate_percent,
                            scale=affine_scale
                        ),
                        iaa.Crop(percent=crop_percent),
                        iaa.Fliplr(0.5),
                    ])
        
        return None    
            
    def __call__(self, img_np_lst, bbs_iaa_lst):
        # transform
        img_np_lst, bbs_iaa_lst = self.seq(images=img_np_lst, bounding_boxes=bbs_iaa_lst)
        bbs_iaa_lst = [bbs_iaa.remove_out_of_image().clip_out_of_image() for bbs_iaa in bbs_iaa_lst]
        
        if self.multiscale_flag:
            target_size = np.random.choice(range(self.multiscale_min, self.multiscale_max, self.multiscale_step))
        else:
            target_size = self.multiscale_max
        resize = iaa.Resize(target_size)

        img_np_lst, bbs_iaa_lst = resize(images=img_np_lst, bounding_boxes=bbs_iaa_lst)

        return img_np_lst, bbs_iaa_lst

class TestAugmentor:
    def __init__(self, imsize):
        self.seq = iaa.Resize(imsize)

    def __call__(self, img_np_lst, bbs_iaa_lst):
        img_np_lst, bbs_iaa_lst = self.seq(images=img_np_lst, bounding_boxes=bbs_iaa_lst)

        return img_np_lst, bbs_iaa_lst

def load_dataset(args):
    if args.train_flag:
        transform = TrainAugmentor(crop_percent      = tuple(args.crop_percent),
                                   color_multiply    = tuple(args.color_multiply), 
                                   translate_percent = args.translate_percent,
                                   affine_scale      = tuple(args.affine_scale),
                                   multiscale_flag   = args.multiscale_flag,
                                   multiscale        = args.multiscale)
    else:
        transform = TestAugmentor(imsize=args.imsize)
        
    dataset = VOC(root              = args.data,
                  transform         = transform,
                  image_set         = 'trainval' if args.train_flag else 'test',
                  pretrained        = args.pretrained,
                  normalized_coord  = args.normalized_coord,
                  num_cls           = args.num_cls)
    return dataset


if __name__ == '__main__':
    from utils import ten2pil

    dataset = VOC(root='./DATA/VOC2028/', 
                  transform=TrainAugmentor(),
                  image_set='trainval', 
                  pretrained='imagenet',
                  normalized_coord=False,
                  num_cls=2)

    dataloader  = torch.utils.data.DataLoader(dataset       = dataset,
                                              batch_size    = 10,
                                              shuffle       = False,
                                              num_workers   = 1,
                                              collate_fn    = dataset.collate_fn)

    image, heatmap, offset, wh, mask, info = next(iter(dataloader))

    image_pil = ten2pil(image, 'imagenet')
    image_pil.save('image.png')

    heatmap_pil_1 = ten2pil(heatmap[:, 0:1, :, :], None)
    heatmap_pil_2 = ten2pil(heatmap[:, 1:2, :, :], None)
    heatmap_pil_1.save('heatmap_0.png')
    heatmap_pil_2.save('heatmap_1.png')
