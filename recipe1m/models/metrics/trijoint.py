import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from . import utils

from tqdm import tqdm 
from recipe1m.models.networks.recipe_networks.networks import AvgPoolSequence
import pickle 

class Trijoint(nn.Module):

    def __init__(self, opt, with_classif=False, engine=None, mode='train'):
        super(Trijoint, self).__init__()
        self.mode = mode
        self.with_classif = with_classif
        self.engine = engine
        # Attributs to process 1000*10 matchs
        # for the retrieval evaluation procedure
        self.nb_bags_retrieval = opt['nb_bags']
        self.nb_matchs_per_bag = opt['nb_matchs_per_bag']
        self.nb_matchs_expected = self.nb_bags_retrieval * self.nb_matchs_per_bag
        self.nb_matchs_saved = 0

        if opt.get('keep_background', False):
            self.ignore_index = None
        else:
            self.ignore_index = 0

        self.identifiers = {'image': [], 'recipe': []}

        if engine and self.mode == 'eval':
            self.split = engine.dataset[mode].split
            engine.register_hook('eval_on_end_epoch', self.calculate_metrics)



    def forward(self, cri_out, net_out, batch):
        out = {}
        if self.with_classif:
            # Accuracy
            [out['acc_image']] = utils.accuracy(net_out['image_classif'].detach().cpu(),
                                             batch['image']['class_id'].detach().squeeze().cpu(),
                                             topk=(1,),
                                             ignore_index=self.ignore_index)
            [out['acc_recipe']] = utils.accuracy(net_out['recipe_classif'].detach().cpu(),
                                              batch['recipe']['class_id'].detach().squeeze().cpu(),
                                              topk=(1,),
                                              ignore_index=self.ignore_index)
        if self.engine and self.mode == 'eval':
            # Retrieval
            # batch_size = len(batch['image']['index'])
            batch_size = len(batch['match'].data) # in case of dataparrallel
            for i in range(batch_size):
                if self.nb_matchs_saved == self.nb_matchs_expected:
                    continue
                if batch['match'].data[i][0] == -1:
                    continue

                identifier = '{}_img_{}'.format(self.split, batch['image']['index'][i])
                utils.save_activation(identifier, net_out['image_embedding'][i].detach().cpu())
                self.identifiers['image'].append(identifier)

                identifier = '{}_rcp_{}'.format(self.split, batch['recipe']['index'][i])
                utils.save_activation(identifier, net_out['recipe_embedding'][i].detach().cpu())
                self.identifiers['recipe'].append(identifier)

                self.nb_matchs_saved += 1   
        
        return out  

    def calculate_metrics(self):
        final_nb_bags = math.floor(self.nb_matchs_saved / self.nb_matchs_per_bag)
        final_matchs_left = self.nb_matchs_saved % self.nb_matchs_per_bag

        if final_nb_bags < self.nb_bags_retrieval:
            log_level = Logger.ERROR if self.split == 'test' else Logger.WARNING
            Logger().log_message('Insufficient matchs ({} saved), {} bags instead of {}'.format(
                self.nb_matchs_saved, final_nb_bags, self.nb_bags_retrieval), log_level=log_level)

        Logger().log_message('Computing retrieval ranking for {} x {} matchs'.format(final_nb_bags,
                                                                                     self.nb_matchs_per_bag))
        list_med_im2recipe = []
        list_med_recipe2im = []
        list_recall_at_1_im2recipe = []
        list_recall_at_5_im2recipe = []
        list_recall_at_10_im2recipe = []
        list_recall_at_1_recipe2im = []
        list_recall_at_5_recipe2im = []
        list_recall_at_10_recipe2im = []

        for i in range(final_nb_bags):
            nb_identifiers_image = self.nb_matchs_per_bag
            nb_identifiers_recipe = self.nb_matchs_per_bag

            distances = np.zeros((nb_identifiers_image, nb_identifiers_recipe), dtype=float)

            # load
            im_matrix = None
            rc_matrix = None
            for j in range(self.nb_matchs_per_bag):
                index = j + i * self.nb_matchs_per_bag

                identifier_image = self.identifiers['image'][index]
                activation_image = utils.load_activation(identifier_image)
                if im_matrix is None:
                    im_matrix = torch.zeros(nb_identifiers_image, activation_image.size(0))
                im_matrix[j] = activation_image

                identifier_recipe = self.identifiers['recipe'][index]
                activation_recipe = utils.load_activation(identifier_recipe)
                if rc_matrix is None:
                    rc_matrix = torch.zeros(nb_identifiers_recipe, activation_recipe.size(0))
                rc_matrix[j] = activation_recipe


            
            distances = fast_distance(im_matrix, rc_matrix)

            im2recipe = np.argsort(distances.numpy(), axis=0)
            recipe2im = np.argsort(distances.numpy(), axis=1)
            
            recall_at_1_recipe2im = 0
            recall_at_5_recipe2im = 0
            recall_at_10_recipe2im = 0
            recall_at_1_im2recipe = 0
            recall_at_5_im2recipe = 0
            recall_at_10_im2recipe = 0
            med_rank_im2recipe = []
            med_rank_recipe2im = []

            for i in range(nb_identifiers_image):
                pos_im2recipe = im2recipe[:,i].tolist().index(i)
                pos_recipe2im = recipe2im[i,:].tolist().index(i)

                if pos_im2recipe == 0:
                    recall_at_1_im2recipe += 1
                if pos_im2recipe <= 4:
                    recall_at_5_im2recipe += 1
                if pos_im2recipe <= 9:
                    recall_at_10_im2recipe += 1

                if pos_recipe2im == 0:
                    recall_at_1_recipe2im += 1
                if pos_recipe2im <= 4:
                    recall_at_5_recipe2im += 1
                if pos_recipe2im <= 9:
                    recall_at_10_recipe2im += 1

                med_rank_im2recipe.append(pos_im2recipe + 1) # other works start ranking from 1 for medR
                med_rank_recipe2im.append(pos_recipe2im + 1)
            
            list_med_im2recipe.append(np.median(med_rank_im2recipe))
            list_med_recipe2im.append(np.median(med_rank_recipe2im))
            list_recall_at_1_im2recipe.append(recall_at_1_im2recipe / nb_identifiers_image)
            list_recall_at_5_im2recipe.append(recall_at_5_im2recipe / nb_identifiers_image)
            list_recall_at_10_im2recipe.append(recall_at_10_im2recipe / nb_identifiers_image)
            list_recall_at_1_recipe2im.append(recall_at_1_recipe2im / nb_identifiers_image)
            list_recall_at_5_recipe2im.append(recall_at_5_recipe2im / nb_identifiers_image)
            list_recall_at_10_recipe2im.append(recall_at_10_recipe2im / nb_identifiers_image)

        out = {}
        out['med_im2recipe_mean'] = np.mean(list_med_im2recipe)
        out['med_recipe2im_mean'] = np.mean(list_med_recipe2im)
        out['recall_at_1_im2recipe_mean'] = np.mean(list_recall_at_1_im2recipe)
        out['recall_at_5_im2recipe_mean'] = np.mean(list_recall_at_5_im2recipe)
        out['recall_at_10_im2recipe_mean'] = np.mean(list_recall_at_10_im2recipe)
        out['recall_at_1_recipe2im_mean'] = np.mean(list_recall_at_1_recipe2im)
        out['recall_at_5_recipe2im_mean'] = np.mean(list_recall_at_5_recipe2im)
        out['recall_at_10_recipe2im_mean'] = np.mean(list_recall_at_10_recipe2im)

        out['med_im2recipe_std'] = np.std(list_med_im2recipe)
        out['med_recipe2im_std'] = np.std(list_med_recipe2im)
        out['recall_at_1_im2recipe_std'] = np.std(list_recall_at_1_im2recipe)
        out['recall_at_5_im2recipe_std'] = np.std(list_recall_at_5_im2recipe)
        out['recall_at_10_im2recipe_std'] = np.std(list_recall_at_10_im2recipe)
        out['recall_at_1_recipe2im_std'] = np.std(list_recall_at_1_recipe2im)
        out['recall_at_5_recipe2im_std'] = np.std(list_recall_at_5_recipe2im)
        out['recall_at_10_recipe2im_std'] = np.std(list_recall_at_10_recipe2im)

        for key, value in out.items():
            Logger().log_value('{}_epoch.metric.{}'.format(self.mode,key), float(value), should_print=True)

        for identifier_image in self.identifiers['image']:
            utils.delete_activation(identifier_image)

        for identifier_recipe in self.identifiers['recipe']:
            utils.delete_activation(identifier_recipe)

        self.identifiers = {'image': [], 'recipe': []}
        self.nb_matchs_saved = 0


class CrossTrijoint(nn.Module):

    def __init__(self, opt, with_classif=False, engine=None, mode='train'):
        super(CrossTrijoint, self).__init__()
        self.mode = mode
        self.with_classif = with_classif
        self.engine = engine
        # Attributs to process 1000*10 matchs
        # for the retrieval evaluation procedure
        self.nb_bags_retrieval = opt['nb_bags']
        self.nb_matchs_per_bag = opt['nb_matchs_per_bag']
        self.nb_matchs_expected = self.nb_bags_retrieval * self.nb_matchs_per_bag
        self.nb_matchs_saved = 0
        self.cross_decoder = Options()['model.network.cross_encoder_params'].get('cross_decoder', False)
        if opt.get('keep_background', False):
            self.ignore_index = None
        else:
            self.ignore_index = 0

        self.identifiers = {'image': [], 'recipe': [], 'image_feat': [], 'recipe_feat': [], 'recipe_feat_sep': [], 
        'recipe_feat_mask': [], 'ids': [], 'img_path': []}

        if engine and self.mode == 'eval':
            self.split = engine.dataset[mode].split
            engine.register_hook('eval_on_end_epoch', self.calculate_metrics_de)
            engine.register_hook('eval_on_end_epoch', self.calculate_metrics)

        if Options()['misc'].get('device_id', False):
            self.device_ids = Options()['misc.device_id']
            if isinstance(self.device_ids, list):
                self.device_id = self.device_ids[0]
            else:
                self.device_id = self.device_ids
            self.device = torch.device('cuda:'+ str(self.device_id) if Options()['misc.cuda'] else 'cpu')
        else:
            self.device = torch.device('cuda' if Options()['misc.cuda'] else 'cpu')

        self.k_test = opt.get('k_test', 10)
        self.cross_decoder_img = Options()['model.network'].get('cross_decoder_image', False)
        self.context_image = Options()['model.network'].get('context_image', 0) # 0 title, 1 ingrds, 2 instrs

        self.save_ids = opt.get('save_ids', False)
        self.save_dir = Options()['exp.dir']
    def forward(self, cri_out, net_out, batch):
        out = {}
        self.recipe_max_len = net_out['recipe_feat'].shape[1]
        if self.with_classif:
            # Accuracy
            [out['acc_image']] = utils.accuracy(net_out['image_classif'].detach().cpu(),
                                             batch['image']['class_id'].detach().squeeze().cpu(),
                                             topk=(1,),
                                             ignore_index=self.ignore_index)
            [out['acc_recipe']] = utils.accuracy(net_out['recipe_classif'].detach().cpu(),
                                              batch['recipe']['class_id'].detach().squeeze().cpu(),
                                              topk=(1,),
                                              ignore_index=self.ignore_index)
        if self.engine and self.mode == 'eval':
            # Retrieval
            batch_size = len(batch['match'].data) 
            for i in range(batch_size):
                if self.nb_matchs_saved == self.nb_matchs_expected:
                    continue
                if batch['match'].data[i][0] == -1:
                    continue

                identifier = '{}_img_{}'.format(self.split, batch['image']['index'][i])
                utils.save_activation(identifier, net_out['image_embedding'][i].detach().cpu())
                self.identifiers['image'].append(identifier)

                identifier_feat = '{}_img_{}_feat'.format(self.split, batch['image']['index'][i])
                utils.save_activation(identifier_feat, net_out['image_feat'][i].detach().cpu())
                self.identifiers['image_feat'].append(identifier_feat)

                identifier = '{}_rcp_{}'.format(self.split, batch['recipe']['index'][i])
                utils.save_activation(identifier, net_out['recipe_embedding'][i].detach().cpu())
                self.identifiers['recipe'].append(identifier)

                identifier_feat = '{}_rcp_{}_feat'.format(self.split, batch['recipe']['index'][i])
                utils.save_activation(identifier_feat, net_out['recipe_feat'][i].detach().cpu())
                self.identifiers['recipe_feat'].append(identifier_feat)

                if self.cross_decoder_img:
                    identifier_feat_sep = '{}_rcp_{}_feat_sep'.format(self.split, batch['recipe']['index'][i])
                    utils.save_activation(identifier_feat_sep, net_out['recipe_feat_sep'][i].detach().cpu())
                    self.identifiers['recipe_feat_sep'].append(identifier_feat_sep)
                if self.save_ids:
                    self.identifiers['ids'].append(batch['recipe']['ids'][i])
                    self.identifiers['img_path'].append(batch['image']['path'][i])


                self.nb_matchs_saved += 1   
        
        return out  

    def calculate_metrics(self):
        final_nb_bags = math.floor(self.nb_matchs_saved / self.nb_matchs_per_bag)
        final_matchs_left = self.nb_matchs_saved % self.nb_matchs_per_bag

        if final_nb_bags < self.nb_bags_retrieval:
            log_level = Logger.ERROR if self.split == 'test' else Logger.WARNING
            Logger().log_message('Insufficient matchs ({} saved), {} bags instead of {}'.format(
                self.nb_matchs_saved, final_nb_bags, self.nb_bags_retrieval), log_level=log_level)

        Logger().log_message('Computing retrieval ranking for {} x {} matchs'.format(final_nb_bags,
                                                                                     self.nb_matchs_per_bag))
        list_med_im2recipe = []
        list_med_recipe2im = []
        list_recall_at_1_im2recipe = []
        list_recall_at_5_im2recipe = []
        list_recall_at_10_im2recipe = []
        list_recall_at_1_recipe2im = []
        list_recall_at_5_recipe2im = []
        list_recall_at_10_recipe2im = []

        for i in range(final_nb_bags):
            nb_identifiers_image = self.nb_matchs_per_bag
            nb_identifiers_recipe = self.nb_matchs_per_bag

            distances = np.zeros((nb_identifiers_image, nb_identifiers_recipe), dtype=float)

            # load
            im_matrix = None
            rc_matrix = None
            im_matrix_feat = None 
            rc_matrix_feat_ = []
            seq_lens = []

            rc_matrix_feat_sep_ = []
            seq_lens_sep = []

            rc_matrix_feat_mask_ = []
            seq_lens_mask = []

            ids = []
            img_path = []
            for j in range(self.nb_matchs_per_bag):
                index = j + i * self.nb_matchs_per_bag

                identifier_image = self.identifiers['image'][index]
                activation_image = utils.load_activation(identifier_image)
                if im_matrix is None:
                    im_matrix = torch.zeros(nb_identifiers_image, activation_image.size(0))
                im_matrix[j] = activation_image

                identifier_image_feat = self.identifiers['image_feat'][index]
                activation_image_feat = utils.load_activation(identifier_image_feat)
                if im_matrix_feat is None:
                    im_matrix_feat = torch.zeros(nb_identifiers_image, activation_image_feat.size(0), activation_image_feat.size(1))
                im_matrix_feat[j] = activation_image_feat


                identifier_recipe = self.identifiers['recipe'][index]
                activation_recipe = utils.load_activation(identifier_recipe)
                if rc_matrix is None:
                    rc_matrix = torch.zeros(nb_identifiers_recipe, activation_recipe.size(0))
                rc_matrix[j] = activation_recipe


                identifier_recipe_feat = self.identifiers['recipe_feat'][index]
                
                activation_recipe_feat = utils.load_activation(identifier_recipe_feat)
                seq_lens.append(activation_recipe_feat.shape[0])

                rc_matrix_feat_.append(activation_recipe_feat)

                if self.cross_decoder_img:
                    identifier_recipe_feat_sep = self.identifiers['recipe_feat_sep'][index]
                    activation_recipe_feat_sep = utils.load_activation(identifier_recipe_feat_sep)
                    seq_lens_sep.append(activation_recipe_feat_sep.shape[0])
                    rc_matrix_feat_sep_.append(activation_recipe_feat_sep)

                if self.save_ids:
                    ids.append(self.identifiers['ids'][index])
                    img_path.append(self.identifiers['img_path'][index])


            rc_matrix_feat = torch.zeros(nb_identifiers_recipe, max(seq_lens), activation_recipe_feat.size(1))
            for j in range(self.nb_matchs_per_bag):
                rc_matrix_feat[j, :rc_matrix_feat_[j].shape[0], :] = rc_matrix_feat_[j]

            if self.cross_decoder_img:
                rc_matrix_feat_sep = torch.zeros(nb_identifiers_recipe, max(seq_lens_sep), activation_recipe_feat_sep.size(1))
                for j in range(self.nb_matchs_per_bag):
                    rc_matrix_feat_sep[j, :rc_matrix_feat_sep_[j].shape[0], :] = rc_matrix_feat_sep_[j]


            distances = fast_distance(im_matrix, rc_matrix) * -1
            score_matrix_i2t = torch.full(distances.size(),-100.0)#.to(self.device)
            for i,sims in enumerate(distances): 
                topk_sim, topk_idx = sims.topk(k=self.k_test, dim=0)
                image_feats = im_matrix_feat[i].repeat(self.k_test,1,1)

                if self.cross_decoder_img:
                    image_feat_pos = self.engine.model.network.cross_decoder_image(image_feats.to(self.device), context=rc_matrix_feat_sep[topk_idx].to(self.device))
                else:
                    image_feat_pos = image_feats

                if self.cross_decoder:
                    output_im2rc = self.engine.model.network.cross_encoder(rc_matrix_feat[topk_idx].to(self.device), context=image_feat_pos.to(self.device))
                else:
                    cross_input_im2rc = torch.cat((image_feat_pos, rc_matrix_feat[topk_idx]), dim=1) # (bs, l1 + l2, dim)
                    output_im2rc = self.engine.model.network.cross_encoder(cross_input_im2rc.to(self.device))

                if len(output_im2rc.size()) > 2 and (self.cross_decoder or not self.engine.model.network.cross_encoder.get_tokens):
                    output_im2rc = output_im2rc.mean(1)


                score = self.engine.model.network.proj_cross(output_im2rc)[:,1] # index=1 for matching 
                score_matrix_i2t[i,topk_idx] = score.detach().cpu()
            distances = distances.t()
            score_matrix_t2i = torch.full(distances.size(),-100.0)#.to(self.device)
            
            
            for i,sims in enumerate(distances): 
                
                topk_sim, topk_idx = sims.topk(k=self.k_test, dim=0)
                image_feats = im_matrix_feat[topk_idx]

                if self.cross_decoder_img:
                    image_feat_pos = self.engine.model.network.cross_decoder_image(image_feats.to(self.device), context=rc_matrix_feat_sep[i].repeat(self.k_test,1,1).to(self.device))
                else:
                    image_feat_pos = image_feats

                if self.cross_decoder:
                    output_rc2im = self.engine.model.network.cross_encoder(rc_matrix_feat[i].repeat(self.k_test,1,1).to(self.device), context=image_feat_pos.to(self.device))
                else:
                    cross_input_rc2im = torch.cat((image_feat_pos, rc_matrix_feat[i].repeat(self.k_test,1,1)), dim=1) # (bs, l1 + l2, dim)
                    output_rc2im = self.engine.model.network.cross_encoder(cross_input_rc2im.to(self.device))

                if len(output_rc2im.size()) > 2 and (self.cross_decoder or not self.engine.model.network.cross_encoder.get_tokens):
                    output_rc2im = output_rc2im.mean(1)


                score = self.engine.model.network.proj_cross(output_rc2im)[:,1] # index=1 for matching 
                score_matrix_t2i[i,topk_idx] = score.detach().cpu()

            im2recipe = np.argsort(-1*score_matrix_i2t.numpy(), axis=1)
            recipe2im = np.argsort(-1*score_matrix_t2i.numpy(), axis=1)
            
            if self.save_ids:
                with open(os.path.join(self.save_dir, 'saved_ids'), 'wb') as fp:
                    pickle.dump(ids, fp)
                with open(os.path.join(self.save_dir, 'img_path'), 'wb') as fp:
                    pickle.dump(img_path, fp)
                    
                np.save(os.path.join(self.save_dir,'im2recipe'), im2recipe)
                np.save(os.path.join(self.save_dir, 'recipe2im'), recipe2im)

            recall_at_1_recipe2im = 0
            recall_at_5_recipe2im = 0
            recall_at_10_recipe2im = 0
            recall_at_1_im2recipe = 0
            recall_at_5_im2recipe = 0
            recall_at_10_im2recipe = 0
            med_rank_im2recipe = []
            med_rank_recipe2im = []

            for i in range(nb_identifiers_image):
                pos_im2recipe = im2recipe[i, :].tolist().index(i)

                pos_recipe2im = recipe2im[i, :].tolist().index(i)

                if pos_im2recipe == 0:
                    recall_at_1_im2recipe += 1
                if pos_im2recipe <= 4:
                    recall_at_5_im2recipe += 1
                if pos_im2recipe <= 9:
                    recall_at_10_im2recipe += 1

                if pos_recipe2im == 0:
                    recall_at_1_recipe2im += 1
                if pos_recipe2im <= 4:
                    recall_at_5_recipe2im += 1
                if pos_recipe2im <= 9:
                    recall_at_10_recipe2im += 1

                med_rank_im2recipe.append(pos_im2recipe + 1) # other works start ranking from 1 for medR
                med_rank_recipe2im.append(pos_recipe2im + 1)
            
            list_med_im2recipe.append(np.median(med_rank_im2recipe))
            list_med_recipe2im.append(np.median(med_rank_recipe2im))
            list_recall_at_1_im2recipe.append(recall_at_1_im2recipe / nb_identifiers_image)
            list_recall_at_5_im2recipe.append(recall_at_5_im2recipe / nb_identifiers_image)
            list_recall_at_10_im2recipe.append(recall_at_10_im2recipe / nb_identifiers_image)
            list_recall_at_1_recipe2im.append(recall_at_1_recipe2im / nb_identifiers_image)
            list_recall_at_5_recipe2im.append(recall_at_5_recipe2im / nb_identifiers_image)
            list_recall_at_10_recipe2im.append(recall_at_10_recipe2im / nb_identifiers_image)

        out = {}
        out['med_im2recipe_mean'] = np.mean(list_med_im2recipe)
        out['med_recipe2im_mean'] = np.mean(list_med_recipe2im)
        out['recall_at_1_im2recipe_mean'] = np.mean(list_recall_at_1_im2recipe)
        out['recall_at_5_im2recipe_mean'] = np.mean(list_recall_at_5_im2recipe)
        out['recall_at_10_im2recipe_mean'] = np.mean(list_recall_at_10_im2recipe)
        out['recall_at_1_recipe2im_mean'] = np.mean(list_recall_at_1_recipe2im)
        out['recall_at_5_recipe2im_mean'] = np.mean(list_recall_at_5_recipe2im)
        out['recall_at_10_recipe2im_mean'] = np.mean(list_recall_at_10_recipe2im)

        out['med_im2recipe_std'] = np.std(list_med_im2recipe)
        out['med_recipe2im_std'] = np.std(list_med_recipe2im)
        out['recall_at_1_im2recipe_std'] = np.std(list_recall_at_1_im2recipe)
        out['recall_at_5_im2recipe_std'] = np.std(list_recall_at_5_im2recipe)
        out['recall_at_10_im2recipe_std'] = np.std(list_recall_at_10_im2recipe)
        out['recall_at_1_recipe2im_std'] = np.std(list_recall_at_1_recipe2im)
        out['recall_at_5_recipe2im_std'] = np.std(list_recall_at_5_recipe2im)
        out['recall_at_10_recipe2im_std'] = np.std(list_recall_at_10_recipe2im)

        for key, value in out.items():
            Logger().log_value('{}_epoch.metric.{}'.format(self.mode,key), float(value), should_print=True)


        for identifier_image in self.identifiers['image']:
            utils.delete_activation(identifier_image)

        for identifier_recipe in self.identifiers['recipe']:
            utils.delete_activation(identifier_recipe)

        for identifier_image in self.identifiers['image_feat']:
            utils.delete_activation(identifier_image)

        for identifier_recipe in self.identifiers['recipe_feat']:
            utils.delete_activation(identifier_recipe)

        if self.cross_decoder_img:
            for identifier_recipe in self.identifiers['recipe_feat_sep']:
                utils.delete_activation(identifier_recipe)



        self.identifiers = {'image': [], 'recipe': [], 'image_feat': [], 'recipe_feat': [], 'recipe_feat_sep': [], 'recipe_feat_mask': []}
        self.nb_matchs_saved = 0

    def calculate_metrics_de(self):
        final_nb_bags = math.floor(self.nb_matchs_saved / self.nb_matchs_per_bag)
        final_matchs_left = self.nb_matchs_saved % self.nb_matchs_per_bag

        if final_nb_bags < self.nb_bags_retrieval:
            log_level = Logger.ERROR if self.split == 'test' else Logger.WARNING
            Logger().log_message('Insufficient matchs ({} saved), {} bags instead of {}'.format(
                self.nb_matchs_saved, final_nb_bags, self.nb_bags_retrieval), log_level=log_level)

        Logger().log_message('Computing retrieval ranking for {} x {} matchs'.format(final_nb_bags,
                                                                                     self.nb_matchs_per_bag))
        list_med_im2recipe = []
        list_med_recipe2im = []
        list_recall_at_1_im2recipe = []
        list_recall_at_5_im2recipe = []
        list_recall_at_10_im2recipe = []
        list_recall_at_1_recipe2im = []
        list_recall_at_5_recipe2im = []
        list_recall_at_10_recipe2im = []

        for i in range(final_nb_bags):
            nb_identifiers_image = self.nb_matchs_per_bag
            nb_identifiers_recipe = self.nb_matchs_per_bag

            distances = np.zeros((nb_identifiers_image, nb_identifiers_recipe), dtype=float)

            # load
            im_matrix = None
            rc_matrix = None
            for j in range(self.nb_matchs_per_bag):
                index = j + i * self.nb_matchs_per_bag

                identifier_image = self.identifiers['image'][index]
                activation_image = utils.load_activation(identifier_image)
                if im_matrix is None:
                    im_matrix = torch.zeros(nb_identifiers_image, activation_image.size(0))
                im_matrix[j] = activation_image

                identifier_recipe = self.identifiers['recipe'][index]
                activation_recipe = utils.load_activation(identifier_recipe)
                if rc_matrix is None:
                    rc_matrix = torch.zeros(nb_identifiers_recipe, activation_recipe.size(0))
                rc_matrix[j] = activation_recipe


            distances = fast_distance(im_matrix, rc_matrix)


            im2recipe = np.argsort(distances.numpy(), axis=0)
            recipe2im = np.argsort(distances.numpy(), axis=1)
            
            recall_at_1_recipe2im = 0
            recall_at_5_recipe2im = 0
            recall_at_10_recipe2im = 0
            recall_at_1_im2recipe = 0
            recall_at_5_im2recipe = 0
            recall_at_10_im2recipe = 0
            med_rank_im2recipe = []
            med_rank_recipe2im = []

            for i in range(nb_identifiers_image):
                pos_im2recipe = im2recipe[:,i].tolist().index(i)
                pos_recipe2im = recipe2im[i,:].tolist().index(i)

                if pos_im2recipe == 0:
                    recall_at_1_im2recipe += 1
                if pos_im2recipe <= 4:
                    recall_at_5_im2recipe += 1
                if pos_im2recipe <= 9:
                    recall_at_10_im2recipe += 1

                if pos_recipe2im == 0:
                    recall_at_1_recipe2im += 1
                if pos_recipe2im <= 4:
                    recall_at_5_recipe2im += 1
                if pos_recipe2im <= 9:
                    recall_at_10_recipe2im += 1

                med_rank_im2recipe.append(pos_im2recipe + 1) # other works start ranking from 1 for medR
                med_rank_recipe2im.append(pos_recipe2im + 1)
            
            list_med_im2recipe.append(np.median(med_rank_im2recipe))
            list_med_recipe2im.append(np.median(med_rank_recipe2im))
            list_recall_at_1_im2recipe.append(recall_at_1_im2recipe / nb_identifiers_image)
            list_recall_at_5_im2recipe.append(recall_at_5_im2recipe / nb_identifiers_image)
            list_recall_at_10_im2recipe.append(recall_at_10_im2recipe / nb_identifiers_image)
            list_recall_at_1_recipe2im.append(recall_at_1_recipe2im / nb_identifiers_image)
            list_recall_at_5_recipe2im.append(recall_at_5_recipe2im / nb_identifiers_image)
            list_recall_at_10_recipe2im.append(recall_at_10_recipe2im / nb_identifiers_image)

        out = {}
        out['de_med_im2recipe_mean'] = np.mean(list_med_im2recipe)
        out['de_med_recipe2im_mean'] = np.mean(list_med_recipe2im)
        out['de_recall_at_1_im2recipe_mean'] = np.mean(list_recall_at_1_im2recipe)
        out['de_recall_at_5_im2recipe_mean'] = np.mean(list_recall_at_5_im2recipe)
        out['de_recall_at_10_im2recipe_mean'] = np.mean(list_recall_at_10_im2recipe)
        out['de_recall_at_1_recipe2im_mean'] = np.mean(list_recall_at_1_recipe2im)
        out['de_recall_at_5_recipe2im_mean'] = np.mean(list_recall_at_5_recipe2im)
        out['de_recall_at_10_recipe2im_mean'] = np.mean(list_recall_at_10_recipe2im)

        out['de_med_im2recipe_std'] = np.std(list_med_im2recipe)
        out['de_med_recipe2im_std'] = np.std(list_med_recipe2im)
        out['de_recall_at_1_im2recipe_std'] = np.std(list_recall_at_1_im2recipe)
        out['de_recall_at_5_im2recipe_std'] = np.std(list_recall_at_5_im2recipe)
        out['de_recall_at_10_im2recipe_std'] = np.std(list_recall_at_10_im2recipe)
        out['de_recall_at_1_recipe2im_std'] = np.std(list_recall_at_1_recipe2im)
        out['de_recall_at_5_recipe2im_std'] = np.std(list_recall_at_5_recipe2im)
        out['de_recall_at_10_recipe2im_std'] = np.std(list_recall_at_10_recipe2im)

        for key, value in out.items():
            Logger().log_value('{}_epoch.metric.{}'.format(self.mode,key), float(value), should_print=True)





# mAP ?
# ConfusionMatrix ?

def fast_distance(A,B):
    # A and B must have norm 1 for this to work for the ranking
    return torch.mm(A,B.t()) * -1

def euclidean_distance_fast(A,B):
    n = A.size(0)
    ZA = (A * A).sum(1)
    ZB = (B * B).sum(1)

    ZA = ZA.expand(n,n)
    ZB = ZB.expand(n,n).t()

    D = torch.mm(B, A.t())
    D.mul_(-2)
    D.add_(ZA).add_(ZB)
    D.sqrt_()
    D.t_()
    return D

def euclidean_distance_slow(A,B):
    n = A.size(0)
    D = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            D[i,j] = torch.dist(A[i], B[j])
    return D
