import sys
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

import click
import torchvision
 




class TransformerDecoder(nn.Module):
    def __init__(self, dim_in=300, n_heads=2, n_layers=2, max_seq_tokens=300, get_tokens=False, first_token=False):
        super(TransformerDecoder, self).__init__()
        encoder_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=n_heads)
        self.encoder = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)
        self.first_token = first_token
        self.cls_token = None 
        self.get_tokens = get_tokens

    def forward(self, x, context, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = x.permute(1, 0, 2)
        context = context.permute(1, 0, 2)
        x = self.encoder(x, context, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        x = x.permute(1, 0, 2)
        if self.cls_token is not None and not self.get_tokens:
            x = x[:, 0]
        if self.first_token:
            x = x[:, 0]
        return x



class Transformer(nn.Module):
    def __init__(self, dim_in=300, n_heads=2, n_layers=2, max_seq_tokens=300, get_tokens=False, first_token=False):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.first_token = first_token
        self.cls_token = None 
        self.get_tokens = get_tokens

    def forward(self, x, ignore_mask=None):
        x = self.encoder(x, src_key_padding_mask=ignore_mask)
        if self.cls_token is not None and not self.get_tokens:
            x = x[:, 0]
        if self.first_token:
            x = x[:, 0]
        return x

class CrossTransformerDecoder(nn.Module):
    def __init__(self, dim_in=300, n_heads=2, n_layers=2, max_seq_tokens=300, get_tokens=False, first_token=False, 
        avg_concat=False, context_1=None, context_2=None, context_3=None, get_tokens_sep=None):
        super().__init__()
        self.cross_encoders_1 = TransformerDecoder(dim_in=dim_in, n_heads=n_heads, n_layers=n_layers, 
            max_seq_tokens=max_seq_tokens, get_tokens=get_tokens, first_token=first_token)

        self.cross_encoders_2 = TransformerDecoder(dim_in=dim_in, n_heads=n_heads, n_layers=n_layers, 
            max_seq_tokens=max_seq_tokens, get_tokens=get_tokens, first_token=first_token)

        self.cross_encoders_3 = TransformerDecoder(dim_in=dim_in, n_heads=n_heads, n_layers=n_layers, 
            max_seq_tokens=max_seq_tokens, get_tokens=get_tokens, first_token=first_token)

        self.avg_concat = avg_concat
        self.get_tokens = get_tokens

        self.context_1 = context_1 # 1 or 2
        self.context_2 = context_2 # 0 or 2
        self.context_3 = context_3 # 0 or 1
        self.get_tokens_sep = get_tokens_sep
        self.recipe_elements = None

    def forward(self, x1 , x2, x3, mask_1=None, mask_2=None, mask_3=None):# (title, ingrds, instrs)
        cat_dim = 1
        xs = (x1 , x2, x3)
        masks = (mask_1, mask_2, mask_3)
        if self.context_1: #title
            context_1 = xs[self.context_1]
            mask_context_1 = masks[self.context_1]
        else:
            context_1 = torch.cat((x2, x3), dim=cat_dim)
            if mask_2 is not None and mask_3 is not None:
                mask_context_1 = torch.cat((mask_2, mask_3), dim=1)
            else:
                mask_context_1 = None
        if self.context_2: #ingrds
            context_2 = xs[self.context_2]
            mask_context_2 = masks[self.context_2]
        else:
            context_2 = torch.cat((x1, x3), dim=cat_dim)
            if mask_1 is not None and mask_3 is not None:
                mask_context_2 = torch.cat((mask_1, mask_3), dim=1)
            else:
                mask_context_2 = None
        if self.context_3: #instrs
            context_3 = xs[self.context_3]
            mask_context_3 = masks[self.context_3]
        else:
            context_3 = torch.cat((x1, x2), dim=cat_dim)
            if mask_1 is not None and mask_2 is not None:
                mask_context_3 = torch.cat((mask_1, mask_2), dim=1)
            else:
                mask_context_3 = None

        x_1 = self.cross_encoders_1(x1, context=context_1)
        x_2 = self.cross_encoders_2(x2, context=context_2)
        x_3 = self.cross_encoders_3(x3, context=context_3)
        if self.avg_concat and self.get_tokens:

            x1 = torch.cat((x_1.mean(1), x_2.mean(1), x_3.mean(1)), dim=1)
        
            x2 = torch.cat((x_1, x_2, x_3), dim=1)

            if self.get_tokens_sep:
                return (x1, x2, (x_1, x_2, x_3))
            else:
                return (x1, x2)
        elif self.avg_concat:

            x = torch.cat((x_1.mean(1), x_2.mean(1), x_3.mean(1)), dim=1)
        else:
            x = torch.cat((x_1, x_2, x_3), dim=1)
        return x  



class HTransformerRecipeEmbedding(nn.Module):
    def __init__(self, opt):
        super(HTransformerRecipeEmbedding, self).__init__()
        if Options()['misc'].get("device_id", False):
            ids = Options()['misc.device_id']
            if isinstance(ids, list):
                ids = ids[0]
            self.device = torch.device("cuda:"+str(ids))
        else:
            self.device = torch.device("cuda")
        Logger()(self.device)
        self.path_ingrs = opt['path_ingrs']
        self.with_ingrs = opt['with_ingrs']
        self.with_instrs = opt['with_instrs']
        self.dim_emb = opt['dim_emb']
        self.activations = opt.get('activations', None)
        self.with_titles = opt.get('with_titles', False)

        self.path_vocab = opt.get('path_vocab', None)

        self.hidden_size = opt.get('hidden_size', 512)
        with open(self.path_vocab,'rb') as f:
            data = pickle.load(f)
        self.embedding = nn.Embedding(len(data), self.hidden_size)

        self.get_tokens =  opt.get('get_tokens', False) 
        self.cls_token = opt.get('cls_token', False)
        self.cls_token_cross = self.cls_token
        self.get_tokens_cross_decoder_recipe = opt.get('get_tokens_cross_decoder_recipe', False)
        self.get_tokens_cross =  self.get_tokens_cross_decoder_recipe
        self.n_layers_cross = opt.get('n_layers_cross', 1)
        self.n_heads_cross = opt.get('n_heads_cross', 2)
        self.cross_encoder = opt.get('cross_encoder', False)

        self.cross_decoder_recipe = opt.get('cross_decoder_recipe', False)
        self.avg_concat = opt.get('avg_concat', False)
        self.cross_encoder = opt.get('cross_encoder', False)

        # modules
        if self.with_ingrs:
            self.dim_ingr_in = self.hidden_size
            self.dim_ingr_out = self.dim_ingr_in
            self.encoder_ingrs = Transformer(dim_in=self.dim_ingr_in, n_heads=opt['n_heads'], 
                n_layers=opt['n_layers'], max_seq_tokens=150, get_tokens=self.get_tokens)
            self.encoder_ingr = Transformer(dim_in=self.dim_ingr_in, n_heads=opt['n_heads_single'], 
                n_layers=opt['n_layers_single'], max_seq_tokens=150)

        if self.with_instrs:
            self.encoder_instrs = Transformer(dim_in=self.hidden_size, n_heads=opt['n_heads'], 
                n_layers=opt['n_layers'], max_seq_tokens=300, 
                get_tokens=self.get_tokens)
            self.encoder_instr = Transformer(dim_in=self.hidden_size, n_heads=opt['n_heads_single'], 
                n_layers=opt['n_layers_single'], max_seq_tokens=300)
        if self.with_titles:
            self.encoder_titles = Transformer(dim_in=self.hidden_size, n_heads=opt['n_heads'], 
                n_layers=opt['n_layers'], max_seq_tokens=50, 
                get_tokens=self.get_tokens)

        self.get_tokens_sep = opt.get('cross_decoder_image', False)
        if self.cross_decoder_recipe:
            context_1 = opt.get('context_title', None)
            context_2 = opt.get('context_ingrds', None)
            context_3 = opt.get('context_instrs', None)
            self.encoder_cross = CrossTransformerDecoder(dim_in=self.hidden_size, n_heads=self.n_heads_cross, 
                n_layers=self.n_layers_cross, max_seq_tokens=100, 
                get_tokens=self.get_tokens_cross, avg_concat=self.avg_concat, context_1=context_1, context_2=context_2, 
                context_3=context_3, get_tokens_sep=self.get_tokens_sep)

        self.fusion = 'cat'
        self.dim_recipe = 0
        if self.with_ingrs:
            self.dim_recipe += self.hidden_size
        if self.with_instrs:
            self.dim_recipe += self.hidden_size
        if self.with_titles:
            self.dim_recipe += self.hidden_size
        if self.dim_recipe == 0:
            Logger()('Ingredients or/and instructions must be embedded "--model.network.with_{ingrs,instrs} True"', Logger.ERROR)
        
        if self.get_tokens:
            if not (self.avg_concat and self.cross_decoder_recipe or self.cross_encoder):
                self.dim_recipe = self.hidden_size

        self.fc = nn.Linear(self.dim_recipe, self.dim_emb)
        self.data_parallel = Options()['misc'].get("data_parallel", False)
        self.recipe_elements = None

    def forward_ingrs_instrs(self, ingrs_out=None, instrs_out=None, titles_out=None):
        if self.cross_decoder_recipe:
            fusion_out = self.encoder_cross(x1=titles_out, x2=ingrs_out,x3=instrs_out)
            if self.avg_concat and self.get_tokens_cross:
                x1 = fusion_out[0]
                x2 = fusion_out[1]
                if self.get_tokens_sep:
                    return (x1, x2, fusion_out[2])
                return (x1, x2)
            elif not self.avg_concat and not self.get_tokens_cross:
                fusion_out = fusion_out.mean(1)
            if not self.cross_encoder:
                x = self.fc(fusion_out)
            else:
                x = fusion_out 
        elif self.with_ingrs and self.with_instrs and self.with_titles:
            if self.cross_encoder:
                if self.cls_token:
                    global_fusion_out = torch.cat([titles_out[:, 0], ingrs_out[:, 0], instrs_out[:, 0]], 1)
                else:
                    global_fusion_out = torch.cat([titles_out.mean(1), ingrs_out.mean(1), instrs_out.mean(1)], 1)
                x_global = global_fusion_out
            if self.fusion == 'cat':
                fusion_out = torch.cat([titles_out, ingrs_out, instrs_out], 1)
            else:
                raise ValueError()
            
            if not self.cross_encoder:
                x = self.fc(fusion_out)
            else:
                x = fusion_out
            if self.cross_encoder:
                return (x_global, x)

        elif self.with_ingrs and self.with_instrs:
            if self.fusion == 'cat':
                fusion_out = torch.cat([ingrs_out, instrs_out], 1)
            else:
                raise ValueError()
            x = self.fc(fusion_out)
        elif self.with_ingrs:
            x = self.fc(ingrs_out)
        elif self.with_instrs:
            x = self.fc(instrs_out)
        if self.activations is not None and not self.cross_encoder:
            for name in self.activations:                
                x = nn.functional.__dict__[name](x)
        return x

    def forward(self, recipe):
        if self.with_ingrs:
            ingrs_out = self.forward_ingrs(recipe['layer1']['ingredients'])
        else:
            ingrs_out = None

        if self.with_instrs:
            instrs_out = self.forward_instrs(recipe['layer1']['instructions'])
        else:
            instrs_out = None
        if self.with_titles:
            titles_out = self.forward_titles(recipe['layer1']['title'])
        else:
            titles_out = None

        x = self.forward_ingrs_instrs(ingrs_out, instrs_out, titles_out)      
        return x

    def forward_ingrs(self, ingrs):
        ingrs_ = ingrs.view(ingrs.size(0)*ingrs.size(1), ingrs.size(2))
        emb_out = self.embedding(ingrs_) 
        hn = self.encoder_ingr(emb_out).mean(-2)
        hn = hn.view(ingrs.size(0), ingrs.size(1), -1)
        hn = self.encoder_ingrs(hn)
        if self.encoder_ingrs.cls_token is None and not self.get_tokens:
            hn = hn.mean(1)
        return hn

    def forward_instrs(self, instrs):
        instrs_ = instrs.view(instrs.size(0)*instrs.size(1), instrs.size(2))
        emb_out = self.embedding(instrs_) 
        hn = self.encoder_instr(emb_out).mean(-2) 
        hn = hn.view(instrs.size(0), instrs.size(1), -1)
        hn = self.encoder_instrs(hn)
        if not self.encoder_instrs.cls_token is None and not self.get_tokens:
            hn = hn.mean(1)
        return hn

    def forward_titles(self, titles):
        emb_out = self.embedding(titles) 
        hn = self.encoder_titles(emb_out)
        if not self.encoder_titles.cls_token is None and not self.get_tokens:
            hn = hn.mean(1)
        return hn



