# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Dataset and dataloader functions
"""

import os
import json
import random
random.seed(1234)
from random import choice
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.utils import get_token_ids, list2Tensors
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing



class Recipe1M(Dataset):
    """Dataset class for Recipe1M

    Parameters
    ----------
    root : string
        Path to Recipe1M dataset.
    transform : (callable, optional)
        A function/transform that takes in a PIL image and returns a transformed version.
    split : string
        Dataset split (train, val, or test).
    max_ingrs : int
        Maximum number of ingredients to use.
    max_instrs : int
        Maximum number of instructions to use.
    max_length_ingrs : int
        Maximum length of ingredient sentences.
    max_length_instrs : int
        Maximum length of instruction sentences.
    text_only_data : bool
        Whether to load paired or text-only samples.
    """

    def __init__(self, root, transform=None, split='train',
                 max_ingrs=20,
                 max_instrs=20,
                 max_length_ingrs=15,
                 max_length_instrs=15,
                 text_only_data=False):

        #load vocabulary
        self.vocab_inv = pickle.load(open('../data/vocab.pkl', 'rb'))
        self.vocab = {}
        for k, v in self.vocab_inv.items():
            if type(v) != str:
                v = v[0]
            self.vocab[v] = k
        # suffix to load text only samples or paired samples
        suf = '_noimages' if text_only_data else ''
        self.data = pickle.load(open(os.path.join(root, 'traindata', split + suf + '.pkl'),
                                     'rb'))
        self.root = root
        self.ids = list(self.data.keys())

        self.split = split
        self.transform = transform

        self.max_ingrs = max_ingrs
        self.max_instrs = max_instrs
        self.max_length_ingrs = max_length_ingrs
        self.max_length_instrs = max_length_instrs

        self.text_only_data = text_only_data

    def __getitem__(self, idx):

        entry = self.data[self.ids[idx]]

        if not self.text_only_data:
            # loading images
            if self.split == 'train':
                # if training, pick an image randomly
                img_name = choice(entry['images'])

            else:
                # if test or val we pick the first image
                img_name = entry['images'][0]

            img_name = '/'.join(img_name[:4])+'/'+img_name
            img = Image.open(os.path.join(self.root, self.split, img_name))
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = None

        title = entry['title']
        ingrs = entry['ingredients']
        instrs = entry['instructions']

        # turn text into indexes
        title = torch.Tensor(get_token_ids(title, self.vocab)[:self.max_length_instrs])
        instrs = list2Tensors([get_token_ids(instr, self.vocab)[:self.max_length_instrs] for instr in instrs[:self.max_instrs]])
        ingrs = list2Tensors([get_token_ids(ingr, self.vocab)[:self.max_length_ingrs] for ingr in ingrs[:self.max_ingrs]])

        return img, title, ingrs, instrs, self.ids[idx]

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

    def get_vocab(self):
        try:
            return self.vocab_inv
        except:
            return None


def pad_input(input):
    """
    creates a padded tensor to fit the longest sequence in the batch
    """
    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        targets = torch.zeros(len(input), max(l)).long()
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
        targets = torch.zeros(len(input), max(n), max(l)).long()
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets


def collate_fn(data):
    """ collate to consume and batchify recipe data
    """

    # Sort a data list by caption length (descending order).
    image, titles, ingrs, instrs, ids = zip(*data)

    if image[0] is not None:
        # Merge images (from tuple of 3D tensor to 4D tensor).
        image = torch.stack(image, 0)
    else:
        image = None
    title_targets = pad_input(titles)
    ingredient_targets = pad_input(ingrs)
    instruction_targets = pad_input(instrs)


    return image, title_targets, ingredient_targets, instruction_targets, ids


def get_loader(root, batch_size, resize, im_size, augment=True,
               split='train', mode='train',
               drop_last=True,
               text_only_data=False):
    """Function to get dataset and dataloader for a data split

    Parameters
    ----------
    root : string
        Path to Recipe1M dataset.
    batch_size : int
        Batch size.
    resize : int
        Image size for resizing (keeps aspect ratio)
    im_size : int
        Image size for cropping.
    augment : bool
        Description of parameter `augment`.
    split : string
        Dataset split (train, val, or test)
    mode : string
        Loading mode (impacts augmentations & random sampling)
    drop_last : bool
        Whether to drop the last batch of data.
    text_only_data : type
        Whether to load text-only or paired samples.

    Returns
    -------
    loader : a pytorch DataLoader
    ds : a pytorch Dataset

    """

    transforms_list = [transforms.Resize((resize))]

    if mode == 'train' and augment:
        # Image preprocessing, normalization for pretrained resnet
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomCrop(im_size))

    else:
        transforms_list.append(transforms.CenterCrop(im_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))

    transforms_ = transforms.Compose(transforms_list)

    ds = Recipe1M(root, transform=transforms_, split=split,
                  text_only_data=text_only_data)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=multiprocessing.cpu_count(),
                        collate_fn=collate_fn, drop_last=drop_last)

    return loader, ds


import os
import lmdb
import pickle
import torch
import torch.utils.data as data

from PIL import Image

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.datasets import utils
from bootstrap.datasets import transforms

from .batch_sampler import BatchSamplerTripletClassif
from bootstrap.lib.options import Options

def default_items_tf():
    return transforms.Compose([
        transforms.ListDictsToDictLists(),
        transforms.PadTensors(value=0),
        transforms.StackTensors()
    ])


class Dataset(data.Dataset):

    def __init__(self, dir_data, split, batch_size, nb_threads, items_tf=default_items_tf):
        super(Dataset, self).__init__()
        self.dir_data = dir_data
        self.split = split
        self.batch_size = batch_size
        self.nb_threads = nb_threads
        self.items_tf = default_items_tf

    def make_batch_loader(self, shuffle=True):
        # allways shuffle even for valset/testset
        # see testing procedure

        if Options()['dataset'].get("debug", False):
            return data.DataLoader(self,
                batch_size=self.batch_size,
                num_workers=self.nb_threads,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.items_tf(),
                drop_last=True) # Removing last batch if not full (quick fix accuracy calculation with class 0 only)
        else:
            return data.DataLoader(self,
                batch_size=self.batch_size,
                num_workers=self.nb_threads,
                shuffle=shuffle,
                pin_memory=True,
                collate_fn=self.items_tf(),
                drop_last=True) # Removing last batch if not full (quick fix accuracy calculation with class 0 only)

class DatasetLMDB(Dataset):

    def __init__(self, dir_data, split, batch_size, nb_threads):
        super(DatasetLMDB, self).__init__(dir_data, split, batch_size, nb_threads)
        self.dir_lmdb = os.path.join(self.dir_data, 'data_lmdb')

        self.path_envs = {}
        self.path_envs['ids'] = os.path.join(self.dir_lmdb, split, 'ids.lmdb')
        self.path_envs['numims'] = os.path.join(self.dir_lmdb, split, 'numims.lmdb')
        self.path_envs['impos'] = os.path.join(self.dir_lmdb, split, 'impos.lmdb')
        self.path_envs['ingrs'] = os.path.join(self.dir_lmdb, split, 'ingrs.lmdb')
        self.path_envs['ilens'] = os.path.join(self.dir_lmdb, split, 'ilens.lmdb')
        self.path_envs['classes'] = os.path.join(self.dir_lmdb, split, 'classes.lmdb')
        self.path_envs['rlens'] = os.path.join(self.dir_lmdb, split, 'rlens.lmdb')
        self.path_envs['rbps'] = os.path.join(self.dir_lmdb, split, 'rbps.lmdb')
        self.path_envs['numims'] = os.path.join(self.dir_lmdb, split, 'numims.lmdb')
        # len(stvecs) train == 2163024
        self.path_envs['stvecs'] = os.path.join(self.dir_lmdb, split, 'stvecs.lmdb')
        # len(imnames) train == 383687
        self.path_envs['imnames'] = os.path.join(self.dir_lmdb, split, 'imnames.lmdb')
        self.path_envs['ims'] = os.path.join(self.dir_lmdb, split, 'ims.lmdb')

        self.envs = {}
        self.envs['ids'] = lmdb.open(self.path_envs['ids'], readonly=True)
        self.envs['classes'] = lmdb.open(self.path_envs['classes'], readonly=True)

        self.txns = {}
        self.txns['ids'] = self.envs['ids'].begin(write=False, buffers=True)
        self.txns['classes'] = self.envs['classes'].begin(write=False, buffers=True)

        self.nb_recipes = self.envs['ids'].stat()['entries']

        self.path_pkl = os.path.join(self.dir_data, 'classes1M.pkl')
        #https://github.com/torralba-lab/im2recipe/blob/master/pyscripts/bigrams.py#L176
        with open(self.path_pkl, 'rb') as f:
            _ = pickle.load(f) # load the first line/object
            self.classes = pickle.load(f) # load the second line/object

        self.cname_to_cid = {v:k for k,v in self.classes.items()}

    def encode(self, value):
        return pickle.dumps(value)

    def decode(self, bytes_value):
        return pickle.loads(bytes_value)

    def get(self, index, env_name):
        buf = self.txns[env_name].get(self.encode(index))
        value = self.decode(bytes(buf))
        return value

    def _load_class(self, index):
        class_id = self.get(index, 'classes') - 1 # lua to python
        return torch.LongTensor([class_id]), self.classes[class_id]

    def __len__(self):
        return self.nb_recipes

    def true_nb_images(self):
        return self.envs['imnames'].stat()['entries']


class Images(DatasetLMDB):

    def __init__(self, dir_data, split, batch_size, nb_threads, image_from='database', image_tf=utils.default_image_tf(256, 224)):
        super(Images, self).__init__(dir_data, split, batch_size, nb_threads)
        self.image_tf = image_tf
        self.dir_img = os.path.join(dir_data,'recipe1M', 'images')

        self.envs['numims'] = lmdb.open(self.path_envs['numims'], readonly=True)
        self.envs['impos'] = lmdb.open(self.path_envs['impos'], readonly=True)
        self.envs['imnames'] = lmdb.open(self.path_envs['imnames'], readonly=True)

        self.txns['numims'] = self.envs['numims'].begin(write=False, buffers=True)
        self.txns['impos'] = self.envs['impos'].begin(write=False, buffers=True)
        self.txns['imnames'] = self.envs['imnames'].begin(write=False, buffers=True)

        self.image_from = image_from
        if self.image_from == 'database':
            self.envs['ims'] = lmdb.open(self.path_envs['ims'], readonly=True)
            self.txns['ims'] = self.envs['ims'].begin(write=False, buffers=True)

    def __getitem__(self, index):
        item = self.get_image(index)
        return item

    def format_path_img(self, raw_path):
        # "recipe1M/images/train/6/b/d/c/6bdca6e490.jpg"
        basename = os.path.basename(raw_path)
        path_img = os.path.join(self.dir_img,
                                self.split,
                                basename[0],
                                basename[1],
                                basename[2],
                                basename[3],
                                basename)
        return path_img

    def get_image(self, index):
        item = {}
        item['data'], item['index'], item['path'] = self._load_image_data(index)
        item['class_id'], item['class_name'] = self._load_class(index)
        return item

    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _load_image_data(self, index):
        # select random image from list of images for that sample
        nb_images = self.get(index, 'numims')
        if Options()['dataset'].get("debug", False):
            im_idx = 0
        else:
            im_idx = torch.randperm(nb_images)[0]
        index_img = self.get(index, 'impos')[im_idx] - 1 # lua to python

        path_img = self.format_path_img(self.get(index_img, 'imnames'))

        if self.image_from == 'pil_loader':
            image_data = self._pil_loader(path_img) 
        elif self.image_from == 'database':
            image_data = self.get(index_img, 'ims')

        if self.image_tf is not None:
            image_data = self.image_tf(image_data)
            
        return image_data, index_img, path_img


class Recipes(DatasetLMDB):

    def __init__(self, dir_data, split, batch_size, nb_threads):
        super(Recipes, self).__init__(dir_data, split, batch_size, nb_threads)
        self.path_ingrs = Options()['model']['network']['path_ingrs']
        with open(self.path_ingrs, 'rb') as fobj:
            data = pickle.load(fobj)
        # idx+1 because 0 is padding
        # https://github.com/torralba-lab/im2recipe/blob/master/pyscripts/mk_dataset.py#L98
        self.ingrid_to_ingrname = {idx+2:name for idx, name in enumerate(data[1])}
        self.ingrid_to_ingrname[1] = '</i>'
        self.ingrname_to_ingrid = {v:k for k,v in self.ingrid_to_ingrname.items()}

        # ~added for visu
        import json
        self.path_layer1 = os.path.join(dir_data, 'recipe1M', 'layer1.json')
        with open(self.path_layer1, 'r') as f:
            self.layer1 = json.load(f)
        self.layer1 = {data['id']:data for data in self.layer1}
        self.envs['ids'] = lmdb.open(self.path_envs['ids'], readonly=True)
        # ~end

        self.envs['ingrs'] = lmdb.open(self.path_envs['ingrs'], readonly=True)
        self.envs['rbps'] = lmdb.open(self.path_envs['rbps'], readonly=True)
        self.envs['rlens'] = lmdb.open(self.path_envs['rlens'], readonly=True)
        # not save length
        self.envs['stvecs'] = lmdb.open(self.path_envs['stvecs'], readonly=True)

        self.txns['ingrs'] = self.envs['ingrs'].begin(write=False, buffers=True)
        self.txns['rbps'] = self.envs['rbps'].begin(write=False, buffers=True)
        self.txns['rlens'] = self.envs['rlens'].begin(write=False, buffers=True)
        self.txns['stvecs'] = self.envs['stvecs'].begin(write=False, buffers=True)

    def __getitem__(self, index):
        item = self.get_recipe(index)
        return item

    def get_recipe(self, index):
        item = {}
        item['class_id'], item['class_name'] = self._load_class(index)
        item['ingrs'] = self._load_ingrs(index)
        item['instrs'] = self._load_instrs(index)
        item['index'] = index
        # ~added for visu
        item['ids'] = self.get(index, 'ids')
        item['layer1'] = self.layer1[item['ids']]
        # ~end
        return item

    def _load_ingrs(self, index):
        ingrs = {}
        ingrs['data'] = torch.LongTensor(self.get(index, 'ingrs'))
        max_length = ingrs['data'].size(0)
        ingrs['lengths'] = max_length - ingrs['data'].eq(0).sum(0).item()
        ingrs['interim'] = self.data_to_words_ingrs(ingrs['data'], ingrs['lengths'])
        return ingrs

    def data_to_words_ingrs(self, data, lengths):
        words = [self.ingrid_to_ingrname[data[i].item()] for i in range(lengths)]
        return words

    def _load_instrs(self, index):
        index_stv = self.get(index, 'rbps') - 1 # -1 cause indexing lua to python
        rlen = self.get(index, 'rlens')
        stvec_size = self.get(index_stv, 'stvecs').size(0)
        instrs = {}
        instrs['data'] = torch.zeros(rlen, stvec_size)
        for i in range(rlen):
            instrs['data'][i] = self.get(index_stv+i, 'stvecs') # -1 cause indexing lua to python
        max_length = instrs['data'].size(0)
        instrs['lengths'] = max_length - instrs['data'][:,0].eq(0).sum(0).item()
        return instrs


class Recipe1M(DatasetLMDB):

    def __init__(self, dir_data, split, batch_size=100, nb_threads=4, freq_mismatch=0.,
            batch_sampler='triplet_classif',
            image_from='dataset', image_tf=utils.default_image_tf(256, 224)):
        super(Recipe1M, self).__init__(dir_data, split, batch_size, nb_threads)
        self.images_dataset = Images(dir_data, split, batch_size, nb_threads, image_from=image_from, image_tf=image_tf)
        self.recipes_dataset = Recipes(dir_data, split, batch_size, nb_threads)
        self.freq_mismatch = freq_mismatch
        self.batch_sampler = batch_sampler

        #self.indices_by_class = self._make_indices_by_class()
        if self.split == 'train' and self.batch_sampler == 'triplet_classif':
            self.indices_by_class = self._make_indices_by_class()

    def _make_indices_by_class(self):
        Logger()('Calculate indices by class...')
        indices_by_class = [[] for class_id in range(len(self.classes))]
        for index in range(len(self.recipes_dataset)):
            class_id = self._load_class(index)[0][0] # bcause (class_id, class_name) and class_id is a Tensor
            indices_by_class[class_id].append(index)
        Logger()('Done!')
        return indices_by_class

    def make_batch_loader(self, shuffle=True):
        if self.split in ['val', 'test'] or self.batch_sampler == 'random':
            if Options()['dataset'].get("debug", False):
                batch_loader = super(Recipe1M, self).make_batch_loader(shuffle=False)
            else:
                batch_loader = super(Recipe1M, self).make_batch_loader(shuffle=shuffle)
            Logger()('Dataset will be sampled with "random" batch_sampler.')
        elif self.batch_sampler == 'triplet_classif':
            batch_sampler = BatchSamplerTripletClassif(
                self.indices_by_class,
                self.batch_size,
                pc_noclassif=0.5,
                nb_indices_same_class=2)
            batch_loader = data.DataLoader(self,
                num_workers=self.nb_threads,
                batch_sampler=batch_sampler,
                pin_memory=True,
                collate_fn=self.items_tf())
            Logger()('Dataset will be sampled with "triplet_classif" batch_sampler.')
        else:
            raise ValueError()
        return batch_loader

    def __getitem__(self, index):
        #ids = self.data['ids'][index]
        item = {}
        item['index'] = index
        item['recipe'] = self.recipes_dataset[index]

        if self.freq_mismatch > 0:
            is_match = torch.rand(1)[0] > self.freq_mismatch
        else:
            is_match = True

        if is_match:
            item['image'] = self.images_dataset[index]
            item['match'] = torch.FloatTensor([1])
        else:
            n_index = int(torch.rand(1)[0] * len(self))
            item['image'] = self.images_dataset[n_index]
            item['match'] = torch.FloatTensor([-1])
        return item

# python -m recipe1m.datasets.recipe1m
if __name__ == '__main__':

    Logger(Options()['logs']['dir'])('lol')


