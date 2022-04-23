from bootstrap.lib.options import Options
from .trijoint import Trijoint
from bootstrap.lib.logger import Logger

def factory(engine=None):

    if Options()['model.name'] == 'trijoint':
        model = Trijoint(
            Options()['model'],
            Options()['dataset.nb_classes'],
            engine.dataset.keys(),
            engine)
    else:
        raise ValueError()
        
    if Options()['misc']['cuda']:
        Logger()('Enabling CUDA mode...')
        if Options()['misc'].get("device_id", False):
            model.cuda(Options()['misc.device_id'])
        else:
            model.cuda()
    return model

