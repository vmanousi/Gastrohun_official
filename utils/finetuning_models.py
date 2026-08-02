# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 08:35:20 2024

@author: Diego Bravo
"""

import pandas as pd
import numpy as np
import os
import torch
from torchinfo import summary
import torchvision.models as models
import torch.nn as nn


def get_discriminative_param_groups(model_ft, trainable_attr, backbone_lr, head_lr):
    """Split a model's currently-trainable parameters into a backbone group
    (whatever is unfrozen outside the head, e.g. the last unfrozen_layers%)
    and a head group (the trainable_attr module: fc/classifier/head/heads),
    so the optimizer can use a smaller LR for the pretrained backbone and a
    larger LR for the head. Only meant for the fine-tuning phase, after
    frozen_* has already set requires_grad on the desired backbone layers."""
    head_module = getattr(model_ft, trainable_attr)
    head_param_ids = {id(p) for p in head_module.parameters()}

    backbone_params = [p for p in model_ft.parameters()
                        if p.requires_grad and id(p) not in head_param_ids]
    head_params = [p for p in head_module.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr})
    return param_groups


def frozen_layers_classifier(model_ft, finetune_pct):
    for param in model_ft.parameters():
        param.requires_grad = True
        
    params = []
    layer_counter = 0
    for (name, module) in model_ft.named_children():
        if name not in ['classifier']:
            for layer in module.children():
                is_trainable = any(param.requires_grad for param in layer.parameters())
                params.append({'Layers Feature': layer , 'No Feature':layer_counter, 'Trainable': is_trainable})
                layer_counter+=1
    df_finetuning = pd.DataFrame()
    df_finetuning = pd.DataFrame(params)
    df_finetuning["Finetuning"] = False
    
    cant_frozen = int((100-finetune_pct) * len(df_finetuning[df_finetuning["Trainable"] == True]) /100)
    cont = 1
    for idx in df_finetuning[df_finetuning["Trainable"] == True].index:
        valor  = df_finetuning["Trainable"].loc[idx]
        if(cont <= cant_frozen):
            df_finetuning.loc[idx,"Finetuning"] = False
        else:
            df_finetuning.loc[idx,"Finetuning"] = True        
        cont = cont+1
    
    layer_counter = 0
    for (name, module) in model_ft.named_children():
        if name not in ['classifier']:
            for layer in module.children():
                for param in layer.parameters():
                    frozen = bool(df_finetuning["Finetuning"].loc[layer_counter])
                    for param in layer.parameters():
                            param.requires_grad = frozen
                layer_counter+=1                
        return model_ft

def frozen_ResNet(model_ft, finetune_pct):
    for param in model_ft.parameters():
        param.requires_grad = True
    
    layer_counter = 0
    params = []
    for (name, module) in model_ft.named_children():
        if name not in ['fc']:
            for layer in module.children():
                #is_trainable = any(param.requires_grad for param in layer.parameters())
                is_trainable = True
                params.append({'Layers Feature': layer , 'No Feature':layer_counter, 'Trainable': is_trainable})
                layer_counter+=1
    df_finetuning = pd.DataFrame(params)
    df_finetuning["Finetuning"] = False
    
    cant_frozen = int((100-finetune_pct) * len(df_finetuning[df_finetuning["Trainable"] == True]) /100)
    if finetune_pct > 95:
        for param in model_ft.conv1.parameters():
            param.requires_grad = True
        for param in model_ft.bn1.parameters():
            param.requires_grad = True
    else:
        for param in model_ft.conv1.parameters():
            param.requires_grad = False
        for param in model_ft.bn1.parameters():
            param.requires_grad = False
            
    
    cont = 1
    for idx in df_finetuning[df_finetuning["Trainable"] == True].index:
        valor  = df_finetuning["Trainable"].loc[idx]
        if(cont <= cant_frozen):
            df_finetuning.loc[idx,"Finetuning"] = False
        else:
            df_finetuning.loc[idx,"Finetuning"] = True        
        cont = cont+1
    
    layer_counter = 0
    for (name, module) in model_ft.named_children():
        if name not in ['fc']:
            for layer in module.children():
                is_trainable = any(param.requires_grad for param in layer.parameters())
                if is_trainable == True:
                    frozen = bool(df_finetuning["Finetuning"].loc[layer_counter])
                    for param in layer.parameters():
                        param.requires_grad = frozen
                layer_counter = layer_counter+1
    return model_ft

     
def frozen_vit(model_ft,finetune_pct):
    for param in model_ft.parameters():
        param.requires_grad = True
    
    params = []
    block_counter = 0
    for layer in model_ft.encoder.layers:
        # Generar el nombre de la capa basado en su índice
        layer_name = f"encoder_layer_{block_counter}"
    
        is_trainable = any(param.requires_grad for param in layer.parameters())
        # Usar el nombre generado en lugar del objeto de la capa directamente
        params.append({'Layer Feature': layer_name, 'No Feature': block_counter, 'Trainable': is_trainable})
        block_counter += 1
    
    df_finetuning = pd.DataFrame(params)
    df_finetuning["Finetuning"] = False
    
    cant_frozen = int((100-finetune_pct) * len(df_finetuning[df_finetuning["Trainable"] == True]) /100)
    if finetune_pct > 95:
        for param in model_ft.conv_proj.parameters():
            param.requires_grad = True
    else:
        for param in model_ft.conv_proj.parameters():
            param.requires_grad = False
    
        
    cont = 1
    for idx in df_finetuning[df_finetuning["Trainable"] == True].index:
        valor  = df_finetuning["Trainable"].loc[idx]
        if(cont <= cant_frozen):
            df_finetuning.loc[idx,"Finetuning"] = False
        else:
            df_finetuning.loc[idx,"Finetuning"] = True        
        cont = cont+1
    
    # Reactivar el entrenamiento solo para los últimos N bloques del encoder 
    for i, block in enumerate(model_ft.encoder.layers):
        for param in block.parameters():
            param.requires_grad = bool(df_finetuning.loc[i,"Finetuning"])
    return model_ft

def frozen_dino(model_ft, finetune_pct):
    """Mirrors frozen_vit's logic (unfreeze the last finetune_pct% of transformer
    blocks, plus the patch-embedding stem only if finetune_pct > 95) but uses
    DINOv2's own module names (backbone.blocks / backbone.patch_embed) instead
    of torchvision ViT's (encoder.layers / conv_proj). model_ft is a
    DinoV2Classifier (see initialize_models.py)."""
    for param in model_ft.parameters():
        param.requires_grad = True

    params = []
    block_counter = 0
    for layer in model_ft.backbone.blocks:
        is_trainable = any(param.requires_grad for param in layer.parameters())
        params.append({'Layer Feature': f"block_{block_counter}", 'No Feature': block_counter, 'Trainable': is_trainable})
        block_counter += 1

    df_finetuning = pd.DataFrame(params)
    df_finetuning["Finetuning"] = False

    cant_frozen = int((100 - finetune_pct) * len(df_finetuning[df_finetuning["Trainable"] == True]) / 100)
    if finetune_pct > 95:
        for param in model_ft.backbone.patch_embed.parameters():
            param.requires_grad = True
    else:
        for param in model_ft.backbone.patch_embed.parameters():
            param.requires_grad = False

    cont = 1
    for idx in df_finetuning[df_finetuning["Trainable"] == True].index:
        if cont <= cant_frozen:
            df_finetuning.loc[idx, "Finetuning"] = False
        else:
            df_finetuning.loc[idx, "Finetuning"] = True
        cont = cont + 1

    for i, block in enumerate(model_ft.backbone.blocks):
        for param in block.parameters():
            param.requires_grad = bool(df_finetuning.loc[i, "Finetuning"])
    return model_ft

def frozen_layers(model_ft, finetune_pct):
    layer_counter = 0
    params = []
    for (name, module) in model_ft.named_children():
        if name == 'features':
            for layer in module.children():
                is_trainable = any(param.requires_grad for param in layer.parameters())
                params.append({'Layers Feature': layer , 'No Feature':layer_counter, 'Trainable': is_trainable})
                layer_counter+=1
    df_finetuning = pd.DataFrame(params)
    df_finetuning["Finetuning"] = False
    
    cant_frozen = int((100-finetune_pct) * len(df_finetuning[df_finetuning["Trainable"] == True]) /100)
    cont = 1
    for idx in df_finetuning[df_finetuning["Trainable"] == True].index:
        valor  = df_finetuning["Trainable"].loc[idx]
        if(cont <= cant_frozen):
            df_finetuning.loc[idx,"Finetuning"] = False
        else:
            df_finetuning.loc[idx,"Finetuning"] = True        
        cont = cont+1
    
    layer_counter = 0
    for (name, module) in model_ft.named_children():
        if name == 'features':
            for layer in module.children():
                is_trainable = any(param.requires_grad for param in layer.parameters())
                if is_trainable == True:
                    frozen = bool(df_finetuning["Finetuning"].loc[layer_counter])
                    for param in layer.parameters():
                        param.requires_grad = frozen
                layer_counter = layer_counter+1
    return model_ft
def frozen_layers_fc(model_ft, finetune_pct):
    layer_counter = 0
    params = []
    for (name, module) in model_ft.named_children():
        if name not in ['fc']:
            for layer in module.children():
                is_trainable = any(param.requires_grad for param in layer.parameters())
                params.append({'Layers Feature': layer , 'No Feature':layer_counter, 'Trainable': is_trainable})
                layer_counter+=1
    df_finetuning = pd.DataFrame(params)
    df_finetuning["Finetuning"] = False
    
    cant_frozen = int((100-finetune_pct) * len(df_finetuning[df_finetuning["Trainable"] == True]) /100)
    cont = 1
    for idx in df_finetuning[df_finetuning["Trainable"] == True].index:
        valor  = df_finetuning["Trainable"].loc[idx]
        if(cont <= cant_frozen):
            df_finetuning.loc[idx,"Finetuning"] = False
        else:
            df_finetuning.loc[idx,"Finetuning"] = True        
        cont = cont+1
    
    layer_counter = 0
    for (name, module) in model_ft.named_children():
        if name not in ['fc']:
            for layer in module.children():
                is_trainable = any(param.requires_grad for param in layer.parameters())
                if is_trainable == True:
                    frozen = bool(df_finetuning["Finetuning"].loc[layer_counter])
                    for param in layer.parameters():
                        param.requires_grad = frozen
                layer_counter = layer_counter+1
    return model_ft         
