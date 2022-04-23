import importlib

from ..lib.options import Options
from ..lib.logger import Logger

from .model import DefaultModel
from .model import SimpleModel
from torch import nn 
import torch
from .networks.data_parallel import DataParallel, DistributedDataParallel
import click

def factory(engine=None, rank=0):

    if Options()['misc']['cuda']:
        if Options()['misc'].get("device_id", False):
            if isinstance(Options()['misc'].get("device_id", False), list):
                Options()['misc.device_id'] = rank


    Logger()('Creating model...')

    if Options()['model'].get('import', False):
        module = importlib.import_module(Options()['model']['import'])
        model = module.factory(engine)

    elif Options()['model']['name'] == 'default':
        model = DefaultModel(engine)

    elif Options()['model']['name'] == 'simple':
        model = SimpleModel(engine)

    else:
        raise ValueError()

    # TODO
    # if data_parallel is not None:
    #     if not cuda:
    #         raise ValueError
    #     model = nn.DataParallel(model).cuda()
    #     model.save = lambda x: x.module.save()  
    if Options()['misc']['cuda']:
        model.is_cuda = True
        Logger()('Enabling CUDA mode...')
        if Options()['misc'].get("device_id", False):
            ids = Options()['misc.device_id']
            Logger()('GPUs:', ids, 'are used')
            if isinstance(ids, list):
                model.to(torch.device('cuda:'+str(ids[0])))
                model = DataParallel(model, device_ids = ids)
                Options()['misc.device_id'] = ids[0] ## data should be mapped to one GPU

                # local_rank = rank
                # torch.cuda.set_device(local_rank) 
                # print(local_rank)
                # Logger()(rank)
                # Logger()(Options()['dataset']['nb_threads'])
                # torch.distributed.init_process_group(                                   
                #     backend='nccl',                                         
                #     init_method='env://',                                   
                #     world_size=len(Options()['misc.device_id']),                              
                #     rank=local_rank                                               
                # )
                # model = DistributedDataParallel(model, device_ids=[local_rank],
                #                                   output_device=local_rank)

                # Options()['misc.device_id'] = None ## data should be mapped to one GPU
                # Logger()(Options()['dataset']['nb_threads'])
            else:
                model.to(torch.device('cuda:'+str(ids)))    
                model.cuda(ids)
        else:
            if Options()['misc'].get("data_parrallel", False):
                model.cuda()
                model = DataParallel(model)
            else:
                model.cuda()
    return model
