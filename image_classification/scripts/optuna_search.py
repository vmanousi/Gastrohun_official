# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search over the Phase-2 (fine-tuning) recipe only.
Phase 1 (warm-up) is trained once per model (via a single subprocess call
to train_image_classification.py --warmup_only, reusing that exact,
already-tested code path -- no duplicated warm-up logic here) and its
checkpoint is cached and reused by every trial. Each trial then loads that
checkpoint, freezes/unfreezes layers with the SAME utils.finetuning_models
functions train_image_classification.py uses, and trains Phase 2 in-process
so Optuna can prune clearly-underperforming trials mid-training via a
custom callback that reports val_f1_macro after every validation epoch.

Search space: lr_finetuning (head LR) and backbone_lr_ratio (backbone LR
as a fraction of head LR, always <= head LR) -- everything else
(unfrozen_layers, gamma/step_size schedule, Phase 1) is held fixed, per the
project's already-established best recipe.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

sys.path.append('../../utils')
from dataset_module_image import CustomDataset
from finetuning_models import (frozen_layers_classifier, frozen_layers_fc, frozen_ResNet,
                                frozen_vit, frozen_dino, get_discriminative_param_groups)
from initialize_models import initialize_model
from train_module_image import ModelTrainer

torch.set_float32_matmul_precision('high')

NORMALIZATION_STATS = {
    'dataset': ([0.5990, 0.3664, 0.2769], [0.2847, 0.2190, 0.1772]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

map_categories = {'A1':0,'L1':1,'P1':2,'G1':3,
                  'A2':4,'L2':5,'P2':6,'G2':7,
                  'A3':8,'L3':9,'P3':10,'G3':11,
                  'A4':12,'L4':13,'P4':14,'G4':15,
                  'A5':16,'L5':17,'P5':18,
                  'A6':19,'L6':20,'P6':21,
                  'OTHERCLASS':22}


def get_args_parser():
    parser = argparse.ArgumentParser('Optuna search over the fine-tuning recipe', add_help=False)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--nb_classes', default=23, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--unfrozen_layers', default=40, type=int)

    # Phase 1 (warm-up) -- fixed, run once, not searched
    parser.add_argument('--num_epochs_warmup', default=10, type=int)
    parser.add_argument('--lr_warmup', default=0.001, type=float)

    # Phase 2 (fine-tuning) hyperparameters NOT being searched -- fixed at
    # the project's established recipe values
    parser.add_argument('--gamma_finetuning', default=0.3, type=float)
    parser.add_argument('--step_size_finetuning', default=5, type=int)

    # Search budget
    parser.add_argument('--n_trials', default=20, type=int)
    parser.add_argument('--search_epochs', default=40, type=int,
                         help="Max epochs per trial during search (a tighter cap than the "
                              "eventual full confirmation run's --num_epochs_finetuning).")
    parser.add_argument('--search_patience', default=5, type=int,
                         help="Early-stopping patience during search trials (shorter than the "
                              "usual 10, to end unpromising trials faster).")

    # Search space bounds
    parser.add_argument('--lr_min', default=1e-4, type=float)
    parser.add_argument('--lr_max', default=3e-3, type=float)
    parser.add_argument('--ratio_min', default=1e-3, type=float,
                         help="Min backbone_lr / head_lr ratio")
    parser.add_argument('--ratio_max', default=1.0, type=float,
                         help="Max backbone_lr / head_lr ratio (1.0 = backbone can move as fast as head)")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=40, type=int)

    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--official_split', required=True, type=str)
    parser.add_argument('--label', default='Complete agreement', type=str)
    parser.add_argument('--normalization', default='dataset', choices=['dataset', 'imagenet'], type=str)

    parser.add_argument('--output_dir', required=True, type=str,
                         help="Base directory for this model's whole Optuna study "
                              "(warm-up checkpoint, trial history, best_params.json).")
    return parser


def build_dataloaders(args):
    data_csv = pd.read_csv(args.official_split, index_col=0)
    data_csv[args.label] = data_csv[args.label].replace(map_categories).astype('Int64')
    data_csv.dropna(subset=[args.label], inplace=True)
    data_csv.reset_index(inplace=True, drop=True)

    norm_mean, norm_std = NORMALIZATION_STATS[args.normalization]
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_data = data_csv[data_csv['set_type'] == 'Train']
    train_dataset = CustomDataset(data=train_data, transform=transform, args=args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    valid_data = data_csv[data_csv['set_type'] == 'Validation']
    valid_dataset = CustomDataset(data=valid_data, transform=transform, args=args)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    elements = data_csv[args.label][data_csv["set_type"] == "Train"].value_counts()
    class_weights = torch.Tensor(compute_class_weight(
        class_weight='balanced', classes=np.arange(0, len(elements), 1),
        y=data_csv[args.label][data_csv['set_type'] == "Train"].values
    )).to(torch.device("cuda"))

    return train_loader, valid_loader, class_weights


def ensure_warmup_checkpoint(args):
    """Runs train_image_classification.py --warmup_only exactly once (reusing
    that already-tested code path, not duplicating it) and caches the
    resulting checkpoint for every trial to reuse."""
    warmup_dir = os.path.join(args.output_dir, "warmup")
    os.makedirs(warmup_dir, exist_ok=True)
    warmup_ckpt = os.path.join(warmup_dir, "warmup_model.pt")
    if os.path.exists(warmup_ckpt):
        print("Reusing cached warm-up checkpoint: {}".format(warmup_ckpt))
        return warmup_ckpt

    print("No cached warm-up checkpoint found -- training it once...")
    cmd = [
        sys.executable, "train_image_classification.py",
        "--model", args.model,
        "--input_size", str(args.input_size),
        "--nb_classes", str(args.nb_classes),
        "--num_epochs_warmup", str(args.num_epochs_warmup),
        "--lr_warmup", str(args.lr_warmup),
        "--num_workers", str(args.num_workers),
        "--batch_size", str(args.batch_size),
        "--data_path", args.data_path,
        "--output_dir", warmup_dir,
        "--official_split", args.official_split,
        "--label", args.label,
        "--normalization", args.normalization,
        "--seed", str(args.seed),
        "--warmup_only",
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(warmup_ckpt):
        raise RuntimeError("Warm-up did not produce the expected checkpoint: {}".format(warmup_ckpt))
    return warmup_ckpt


def load_warmed_up_model(args, warmup_ckpt_path, device):
    model_ft, CNN_family = initialize_model(args.model, args.nb_classes, True, (args.input_size, args.input_size))
    for param in model_ft.parameters():
        param.requires_grad = False
    trainable_attr = None
    for attr_name in ['fc', 'classifier', 'head', 'heads']:
        if hasattr(model_ft, attr_name):
            trainable_attr = attr_name
            break
    if trainable_attr is None:
        raise AttributeError("The model does not have any of the expected attributes for training.")
    for param in getattr(model_ft, trainable_attr).parameters():
        param.requires_grad = True
    model_ft.load_state_dict(torch.load(warmup_ckpt_path, map_location=device))
    return model_ft.to(device), CNN_family, trainable_attr


def freeze_for_finetuning(model_ft, CNN_family, unfrozen_layers):
    if CNN_family in ["MaxVit", "SwinTransformer"]:
        return frozen_layers_fc(model_ft, unfrozen_layers)
    elif CNN_family in ["ConvNeXt", "VGG"]:
        return frozen_layers_classifier(model_ft, unfrozen_layers)
    elif CNN_family == "VisionTransformer":
        return frozen_vit(model_ft, unfrozen_layers)
    elif CNN_family in ["DINOv2", "DINO"]:
        return frozen_dino(model_ft, unfrozen_layers)
    elif CNN_family in ["ResNet", "Wide ResNet"]:
        return frozen_ResNet(model_ft, unfrozen_layers)
    else:
        return frozen_layers_fc(model_ft, unfrozen_layers)


class OptunaPruningCallback(pl.Callback):
    """Reports val_f1_macro to Optuna after every validation epoch and
    raises TrialPruned if the pruner decides this trial isn't promising.
    Also tracks the best score seen (not just the last epoch's), matching
    the same 'best checkpoint' semantics ModelCheckpoint uses elsewhere in
    this project -- important because EarlyStopping means the last epoch
    is typically `patience` epochs worse than the true peak."""
    def __init__(self, trial, monitor):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.best_score = float("-inf")

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        current_score = float(current_score)
        if current_score > self.best_score:
            self.best_score = current_score
        self.trial.report(current_score, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(trial, args, warmup_ckpt_path, train_loader, valid_loader, class_weights, device):
    lr_finetuning = trial.suggest_float("lr_finetuning", args.lr_min, args.lr_max, log=True)
    ratio = trial.suggest_float("backbone_lr_ratio", args.ratio_min, args.ratio_max, log=True)
    backbone_lr = lr_finetuning * ratio

    model_ft, CNN_family, trainable_attr = load_warmed_up_model(args, warmup_ckpt_path, device)
    model_ft = freeze_for_finetuning(model_ft, CNN_family, args.unfrozen_layers)

    param_groups = get_discriminative_param_groups(
        model_ft, trainable_attr, backbone_lr=backbone_lr, head_lr=lr_finetuning,
    )

    trainer_module = ModelTrainer(
        model=model_ft, num_classes=args.nb_classes, class_weights=class_weights,
        learning_rate=lr_finetuning, gamma=args.gamma_finetuning,
        step_size=args.step_size_finetuning, param_groups=param_groups,
    ).to(device)

    pruning_callback = OptunaPruningCallback(trial, monitor="val_f1_macro")
    early_stopping = EarlyStopping(monitor="val_f1_macro", patience=args.search_patience,
                                    mode="max", verbose=False)

    trainer = Trainer(
        max_epochs=args.search_epochs,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator="gpu" if torch.cuda.is_available() else None,
        logger=False,
        check_val_every_n_epoch=1,
        callbacks=[early_stopping, pruning_callback],
        enable_checkpointing=False,  # search trials are throwaway -- the winning config gets a
                                      # proper full-budget confirmation run afterward, which is
                                      # the one whose checkpoint actually gets kept
        num_sanity_val_steps=0,
    )
    trainer.fit(trainer_module, train_loader, valid_loader)

    trial.set_user_attr("lr_finetuning", lr_finetuning)
    trial.set_user_attr("backbone_lr_finetuning", backbone_lr)
    trial.set_user_attr("backbone_lr_ratio", ratio)

    if pruning_callback.best_score == float("-inf"):
        raise optuna.TrialPruned()
    return pruning_callback.best_score


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not os.path.exists(args.data_path):
        print("The data path does not exist: {}".format(args.data_path))
        sys.exit(1)
    if not os.path.exists(args.official_split):
        print("The official split does not exist: {}".format(args.official_split))
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building dataloaders...")
    train_loader, valid_loader, class_weights = build_dataloaders(args)

    print("Ensuring warm-up checkpoint exists...")
    warmup_ckpt_path = ensure_warmup_checkpoint(args)

    print("Starting Optuna study: {} trials, search_epochs={}, search_patience={}".format(
        args.n_trials, args.search_epochs, args.search_patience))
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                 study_name="{}_finetune_search".format(args.model))
    study.optimize(
        lambda trial: objective(trial, args, warmup_ckpt_path, train_loader, valid_loader, class_weights, device),
        n_trials=args.n_trials,
    )

    print("="*40)
    print("Best trial: #{}".format(study.best_trial.number))
    print("Best val_f1_macro: {:.4f}".format(study.best_value))
    print("Best params: {}".format(study.best_params))
    print("="*40)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(args.output_dir, "optuna_trials.csv"), index=False)

    best_config = {
        "model": args.model,
        "best_trial_number": study.best_trial.number,
        "best_val_f1_macro": study.best_value,
        "lr_finetuning": study.best_params["lr_finetuning"],
        "backbone_lr_ratio": study.best_params["backbone_lr_ratio"],
        "backbone_lr_finetuning": study.best_params["lr_finetuning"] * study.best_params["backbone_lr_ratio"],
        "unfrozen_layers": args.unfrozen_layers,
        "n_trials": args.n_trials,
        "search_epochs": args.search_epochs,
        "search_patience": args.search_patience,
    }
    with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
        json.dump(best_config, f, indent=2)
    print("Saved best_params.json to {}".format(args.output_dir))
