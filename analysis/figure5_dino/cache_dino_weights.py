import torch

for m in ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8']:
    print('Loading', m)
    torch.hub.load('facebookresearch/dino:main', m)

for m in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']:
    print('Loading', m)
    torch.hub.load('facebookresearch/dinov2', m)

print('All cached.')