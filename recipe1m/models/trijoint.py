import torch
import torch.nn as nn

from bootstrap.lib.options import Options
from bootstrap.datasets import transforms
from bootstrap.models.model import Model

from . import networks
from . import criterions
from . import metrics

from bootstrap.lib.logger import Logger
import os 

class Trijoint(Model):

    def __init__(self,
                 opt,
                 nb_classes,
                 modes=['train', 'eval'],
                 engine=None,
                 cuda_tf=transforms.ToCuda):
        super(Trijoint, self).__init__(engine, cuda_tf=cuda_tf)
        if Options()['misc'].get("device_id", False):
            self.device_id = Options()['misc.device_id']

        self.cross_encoder_model = Options()['model.network'].get('cross_encoder', False)

        self.network = networks.CrossTrijoint(
            opt['network'],
            nb_classes,
            with_classif=opt['with_classif'])


        self.criterions = {}
        self.metrics = {}


        self.itm_loss_weight = Options()['model.criterion'].get('itm_loss_weight', 0)
        if self.itm_loss_weight > 0:
            self.trijoint_metric = Options()['model.metric'].get('trijoint', False) 
        else: 
            self.trijoint_metric = True
        if 'train' in modes:
            self.criterions['train'] = criterions.Trijoint(
                opt['criterion'],
                nb_classes,
                opt['network']['dim_emb'],
                with_classif=opt['with_classif'],
                engine=engine)

            if self.cross_encoder_model and not self.trijoint_metric:
                self.metrics['train'] = metrics.CrossTrijoint(
                opt['metric'],
                with_classif=opt['with_classif'],
                engine=engine,
                mode='train')
            else:
                self.metrics['train'] = metrics.Trijoint(
                    opt['metric'],
                    with_classif=opt['with_classif'],
                    engine=engine,
                    mode='train')

        if 'eval' in modes:
            if self.cross_encoder_model and not self.trijoint_metric:
                self.metrics['eval'] = metrics.CrossTrijoint(
                    opt['metric'],
                    with_classif=opt['with_classif'],
                    engine=engine,
                    mode='eval')
            else:
                self.metrics['eval'] = metrics.Trijoint(
                    opt['metric'],
                    with_classif=opt['with_classif'],
                    engine=engine,
                    mode='eval')



