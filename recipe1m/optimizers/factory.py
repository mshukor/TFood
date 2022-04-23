from bootstrap.lib.options import Options
from .trijoint import Trijoint, TrijointCLIP

def factory(model, engine=None):

    if Options()['optimizer']['name'] == 'trijoint_fixed_fine_tune':
        optimizer = Trijoint(Options()['optimizer'], model, engine)
    else:
        raise ValueError()

    return optimizer

