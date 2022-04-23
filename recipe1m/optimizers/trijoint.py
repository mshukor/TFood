import torch
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from torch.nn.utils.clip_grad import clip_grad_norm_

class Trijoint(torch.optim.Optimizer):

    def __init__(self, opt, model, engine=None):
        self.model = model
        self.lr = opt['lr']
        if opt.get('lr_img', False):
            self.lr_img = opt['lr_img']
        else:
            self.lr_img = self.lr

        if opt.get('lr_rec', False):
            self.lr_rec = opt['lr_rec']
        else:
            self.lr_rec = self.lr
            
        if opt.get('lr_cross', False):
            self.lr_cross = opt['lr_cross']
        else:
            self.lr_cross = self.lr_rec
        self.switch_epoch = opt['switch_epoch']
        self.clip_grad = opt.get('clip_grad', False)
        self.optimizers = {}
        self.optimizers['recipe'] = torch.optim.Adam(self.model.network.get_parameters_recipe(), self.lr_rec)
        self.optimizers['image'] = torch.optim.Adam(self.model.network.get_parameters_image(), self.lr_img)

        self.lr_scheduler_img = opt.get('lr_scheduler_img', False)
        self.lr_scheduler_rec = opt.get('lr_scheduler_rec', False)
        self.lr_scheduler_cross = opt.get('lr_scheduler_cross', False)

        if self.lr_scheduler_img:
            self.scheduler_img = torch.optim.lr_scheduler.StepLR(self.optimizers['image'], step_size=60, gamma=0.316)
        else:
            self.scheduler_img = None

        if self.lr_scheduler_rec:
            self.scheduler_rec = torch.optim.lr_scheduler.StepLR(self.optimizers['recipe'], step_size=60, gamma=0.316)
        else:
            self.scheduler_rec = None

        if Options()['model.network'].get('cross_encoder_params', False):
            if Options()['model.network.cross_encoder_params'].get('cross_optimizer', False):
                self.cross_optimizer = True
                self.optimizers['cross'] = torch.optim.Adam(self.model.network.get_parameters_cross(), self.lr_cross)
                if self.lr_scheduler_cross:
                    self.scheduler_cross = torch.optim.lr_scheduler.StepLR(self.optimizers['cross'], step_size=60, gamma=0.316)
                else:
                    self.scheduler_cross = None
            else:
                self.cross_optimizer = False
                self.scheduler_cross_ = None
        else:
            self.cross_optimizer = False
            self.scheduler_cross_ = None

        if self.lr_scheduler_img or self.lr_scheduler_rec or self.lr_scheduler_cross:
            engine.register_hook('train_on_end_epoch', self.lr_scheduler_step)

        self.current_optimizer_name = 'recipe'
        self.epoch = 0

        cross_encoder_model = Options()['model.network'].get('cross_encoder', False)

        self.freeze_im = Options()['model.network'].get('freeze_im', False)
        self.freeze_rec = Options()['model.network'].get('freeze_rec', False)

        self.lr_clip_decay_rec = Options()['model.network'].get('lr_decay_rec', False)
        if self.lr_clip_decay_rec:
            self.scheduler_rec =  torch.optim.lr_scheduler.StepLR(self.optimizers['recipe'], step_size=1, gamma=0.1)

        self.lr_clip_decay_im = Options()['model.network'].get('lr_decay_im', False)
        if self.lr_clip_decay_im:
            self.scheduler_im =  torch.optim.lr_scheduler.StepLR(self.optimizers['image'], step_size=1, gamma=0.1)

        if self.cross_optimizer:
            self.lr_clip_decay_cross = Options()['model.network'].get('lr_decay_cross', False)
            if self.lr_clip_decay_cross:
                self.lr_decay_cross_w = Options()['model.network'].get('lr_decay_cross_w', 0.1)
                self.scheduler_cross =  torch.optim.lr_scheduler.StepLR(self.optimizers['cross'], step_size=1, gamma=self.lr_decay_cross_w)

        self._activate_model()

        if engine:
            engine.register_hook('train_on_start_epoch', self._auto_fixed_fine_tune)
            if self.clip_grad:
                engine.register_hook('train_on_print', self.print_total_norm)



    def state_dict(self):
        state = {}
        state['optimizers'] = {}
        for key, value in self.optimizers.items():
            state['optimizers'][key] = value.state_dict()
        state['attributs'] = {
            'current_optimizer_name': self.current_optimizer_name,
            'epoch': self.epoch
        }
        return state

    def load_state_dict(self, state_dict):
        for key, _ in self.optimizers.items():
            value = state_dict['optimizers'][key]
            if len(value['state']) != 0: # bug pytorch??
                self.optimizers[key].load_state_dict(value)
        if 'attributs' in state_dict:
            for key, value in state_dict['attributs'].items():
                setattr(self, key, value)

        self._activate_model()



    def zero_grad(self):
        for name in self.current_optimizer_name.split('&'):
            self.optimizers[name].zero_grad()

        if self.cross_optimizer:
            self.optimizers['cross'].zero_grad()
    def step(self, closure=None):
        if self.clip_grad:
            self.clip_grad_norm()
        for name in self.current_optimizer_name.split('&'):
            self.optimizers[name].step(closure)
        if self.cross_optimizer:
            self.optimizers['cross'].step(closure)


    def clip_grad_norm(self):
        params = []
        for k in self.optimizers:
            for group in self.optimizers[k].param_groups:
                for p in group['params']:
                    params.append(p)
        self.total_norm = clip_grad_norm_(params, self.clip_grad)
        Logger().log_value('optimizer.total_norm', self.total_norm.item(), should_print=False)

    def print_total_norm(self):
        Logger()('{}  total_norm: {:.6f}'.format(' '*len('train'), self.total_norm))

    def _activate_model(self):
        optim_name = self.current_optimizer_name
        activate_recipe = (optim_name == 'recipe') or (optim_name == 'recipe&image')
        activate_image = (optim_name == 'image') or (optim_name == 'recipe&image')
        activate_cross = True
        if not self.freeze_rec:
            for p_dict in self.model.network.get_parameters_recipe():
                for p in p_dict['params']:
                    p.requires_grad = activate_recipe
        if not self.freeze_im:
            for p in self.model.network.get_parameters_image():
                p.requires_grad = activate_image
        if self.cross_optimizer:
            for p_dict in self.model.network.get_parameters_cross():
                for p in p_dict['params']:
                    p.requires_grad = activate_cross

    def _auto_fixed_fine_tune(self):
        if self.current_optimizer_name == 'recipe' and self.epoch == self.switch_epoch:
            self.current_optimizer_name = 'recipe&image'
            self.freeze = False
            self.freeze_im = False 
            self.freeze_rec = False
            self._activate_model()
            Logger()('Switched to optimizer '+self.current_optimizer_name)
            if self.lr_clip_decay_rec:
                self.scheduler_rec.step()
            if self.lr_clip_decay_im:
                self.scheduler_im.step()
            if self.cross_optimizer:
                if self.lr_clip_decay_cross:
                    # print(self.scheduler_cross.get_lr())
                    self.scheduler_cross.step()
                    # print(self.scheduler_cross.get_lr())
            

        Logger().log_value('optimizer.is_optimizer_recipe&image',
                           int(self.current_optimizer_name == 'recipe&image'),
                           should_print=False)
        self.epoch += 1

    def lr_scheduler_step(self,):
        if self.lr_scheduler_img:
            self.scheduler_img.step()
        if self.lr_scheduler_rec:
            self.scheduler_rec.step()
        if self.scheduler_cross_:
            self.scheduler_cross_.step()





