"""
Training script per RadioDino + HybridEncoder su MedSegBench.

Versione potenziata che usa lo stesso HybridEncoder di DEIMv2
per colmare il gap di performance.

Supporta lista di modelli nel YAML per eseguire esperimenti multipli.
"""
import os
import sys
import json
import time
import re
import argparse
import csv
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.data.medseg_png_dataset import MedSegPNGDataset
from code.dense.engine.deim.omnirad_seg_model import OmniRadHybridSegModel


# ============================================================================
# METRICS
# ============================================================================

def count_parameters(model: nn.Module, print_details: bool = True) -> Dict[str, Dict[str, int]]:
    """
    Count total and trainable parameters for each component and the whole model.
    
    Returns:
        Dict with component names as keys and {"total": N, "trainable": M} as values.
    """
    param_stats = {}
    
    # Define component mappings
    components = {
        "backbone": "backbone",
        "sta (spatial_prior)": "spatial_prior",
        "hybrid_encoder": "hybrid_encoder", 
        "decoder": "decoder",
        "seg_head": "seg_head",
    }
    
    total_params = 0
    total_trainable = 0
    
    for comp_name, attr_name in components.items():
        if hasattr(model, attr_name):
            module = getattr(model, attr_name)
            if module is not None:
                comp_total = sum(p.numel() for p in module.parameters())
                comp_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                param_stats[comp_name] = {"total": comp_total, "trainable": comp_trainable}
                total_params += comp_total
                total_trainable += comp_trainable
    
    # Add any remaining parameters not in named components
    accounted = sum(s["total"] for s in param_stats.values())
    all_params = sum(p.numel() for p in model.parameters())
    all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if all_params > accounted:
        param_stats["other"] = {
            "total": all_params - accounted,
            "trainable": all_trainable - total_trainable
        }
    
    param_stats["TOTAL"] = {"total": all_params, "trainable": all_trainable}
    
    if print_details:
        print("\n" + "="*60)
        print("MODEL PARAMETERS SUMMARY")
        print("="*60)
        print(f"{'Component':<25} {'Total':>12} {'Trainable':>12} {'Frozen':>12}")
        print("-"*60)
        for comp, stats in param_stats.items():
            frozen = stats["total"] - stats["trainable"]
            if comp == "TOTAL":
                print("-"*60)
            print(f"{comp:<25} {stats['total']:>12,} {stats['trainable']:>12,} {frozen:>12,}")
        print("="*60 + "\n")
    
    return param_stats


def compute_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> Dict[str, float]:
    """
    Computes comprehensive segmentation metrics.
    
    Returns:
        Dict with: mIoU, Dice, Precision, Recall, Specificity, Accuracy, F1
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]
    
    # Per-class metrics
    ious, dices, precisions, recalls, specificities = [], [], [], [], []
    
    total_correct = 0
    total_pixels = preds.numel()
    
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
            
        pred_c = (preds == c)
        targ_c = (targets == c)
        
        tp = (pred_c & targ_c).sum().item()
        fp = (pred_c & ~targ_c).sum().item()
        fn = (~pred_c & targ_c).sum().item()
        tn = (~pred_c & ~targ_c).sum().item()
        
        total_correct += tp + tn
        
        # IoU
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / (union + 1e-6))
        
        # Dice (F1 for segmentation)
        dice_denom = 2 * tp + fp + fn
        if dice_denom > 0:
            dices.append((2 * tp) / (dice_denom + 1e-6))
        
        # Precision
        if tp + fp > 0:
            precisions.append(tp / (tp + fp + 1e-6))
        
        # Recall (Sensitivity)
        if tp + fn > 0:
            recalls.append(tp / (tp + fn + 1e-6))
        
        # Specificity
        if tn + fp > 0:
            specificities.append(tn / (tn + fp + 1e-6))
    
    # Compute means
    metrics = {
        "mIoU": float(sum(ious) / len(ious)) if ious else 0.0,
        "Dice": float(sum(dices) / len(dices)) if dices else 0.0,
        "Precision": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "Recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "Specificity": float(sum(specificities) / len(specificities)) if specificities else 0.0,
        "Accuracy": float(total_correct / (total_pixels * num_classes + 1e-6)) if total_pixels > 0 else 0.0,
    }
    # F1 = Dice for binary, but compute from precision/recall for consistency
    if metrics["Precision"] + metrics["Recall"] > 0:
        metrics["F1"] = 2 * metrics["Precision"] * metrics["Recall"] / (metrics["Precision"] + metrics["Recall"] + 1e-6)
    else:
        metrics["F1"] = 0.0
    
    return metrics


def compute_mean_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Legacy function for backward compatibility."""
    metrics = compute_segmentation_metrics(preds, targets, num_classes, ignore_index)
    return metrics["mIoU"]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=use_amp):
            out = model(images)
            logits = out["pred_masks"]

            if logits.shape[-2:] != masks.shape[-2:]:
                masks_resized = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=logits.shape[-2:],
                    mode="nearest",
                ).long().squeeze(1)
            else:
                masks_resized = masks

            loss = F.cross_entropy(logits, masks_resized)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model and return loss + all metrics.
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0
    # Accumulate metrics
    accumulated_metrics = {
        "mIoU": 0.0, "Dice": 0.0, "Precision": 0.0, 
        "Recall": 0.0, "Specificity": 0.0, "Accuracy": 0.0, "F1": 0.0
    }
    n_batches = 0

    for images, masks in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        with autocast(device_type='cuda', enabled=use_amp):
            out = model(images)
            logits = out["pred_masks"]

            if logits.shape[-2:] != masks.shape[-2:]:
                masks_resized = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=logits.shape[-2:],
                    mode="nearest",
                ).long().squeeze(1)
            else:
                masks_resized = masks

            loss = F.cross_entropy(logits, masks_resized)
        
        preds = logits.argmax(dim=1)
        
        # Compute all metrics
        batch_metrics = compute_segmentation_metrics(preds, masks_resized, num_classes=num_classes)

        running_loss += float(loss.item())
        for k in accumulated_metrics:
            accumulated_metrics[k] += batch_metrics[k]
        n_batches += 1

    if n_batches == 0:
        return 0.0, accumulated_metrics
    
    avg_loss = running_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in accumulated_metrics.items()}
    
    return avg_loss, avg_metrics


def train_on_dataset(dataset_name: str, args: argparse.Namespace) -> Dict | None:
    """
    Train on a single dataset. Returns None if already completed (final_results.json exists).
    """
    # Check if already completed
    run_dir = os.path.join(args.output_dir, f"radio_dino_hybrid_{dataset_name}")
    final_results_path = os.path.join(run_dir, "final_results.json")
    
    if os.path.isfile(final_results_path):
        print(f"[SKIP] {dataset_name} already completed, loading results from {final_results_path}")
        with open(final_results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    image_size = tuple(args.image_size) if args.image_size is not None else (518, 518)

    train_set = MedSegPNGDataset(root=args.root, dataset_name=dataset_name, split="train", image_size=image_size)
    val_set = MedSegPNGDataset(root=args.root, dataset_name=dataset_name, split="val", image_size=image_size)
    test_set = MedSegPNGDataset(root=args.root, dataset_name=dataset_name, split="test", image_size=image_size)

    num_classes = train_set.num_classes

    # Configurazione
    use_sta = getattr(args, "use_sta", True)
    conv_inplane = getattr(args, "conv_inplane", 16)
    hidden_dim = getattr(args, "hidden_dim", None)
    out_indices = getattr(args, "out_indices", (2, 5, 8, 11))
    if isinstance(out_indices, list):
        out_indices = tuple(out_indices)

    # HybridEncoder config
    use_hybrid_encoder = getattr(args, "use_hybrid_encoder", True)
    encoder_cfg = getattr(args, "HybridEncoder", None)
    
    # Se c'è backbone_config, carica HybridEncoder da lì
    backbone_config = getattr(args, "backbone_config", None)
    if backbone_config and os.path.isfile(backbone_config):
        import yaml
        with open(backbone_config, "r", encoding="utf-8") as f:
            bb_cfg = yaml.safe_load(f)
        if "HybridEncoder" in bb_cfg:
            encoder_cfg = bb_cfg["HybridEncoder"]
            print(f"[INFO] Loaded HybridEncoder config from {backbone_config}")

    # Modello RadioDino + HybridEncoder
    model = RadioDinoHybridSegModel(
        model_name=args.model_name,
        num_classes=num_classes,
        image_size=image_size,
        use_sta=use_sta,
        conv_inplane=conv_inplane,
        hidden_dim=hidden_dim,
        out_indices=out_indices,
        use_hybrid_encoder=use_hybrid_encoder,
        encoder_cfg=encoder_cfg,
        freeze_backbone=getattr(args, "freeze_backbone", False),
    ).to(device)
    
    # Freeze backbone if requested
    freeze_backbone = getattr(args, "freeze_backbone", False)
    if freeze_backbone and hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print(f"[INFO] Backbone frozen: {args.model_name}")
    
    # Print parameter counts per component
    count_parameters(model, print_details=True)

    # DataLoaders con persistent_workers per train/val, 0 workers per test
    num_workers = args.workers if args.workers > 0 else 0
    use_persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        #pin_memory=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        #pin_memory=True,
        persistent_workers=use_persistent,
    )
    # Test loader con 0 workers (no persistent)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        #pin_memory=True,
    )

    # Optimizer: only trainable parameters (excludes frozen backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    print(f"[INFO] Optimizer: {len(trainable_params)} parameter groups")

    # AMP (Automatic Mixed Precision)
    use_amp = getattr(args, "use_amp", False) and device.type == "cuda"
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[INFO] Using Automatic Mixed Precision (AMP)")

    # run_dir già definito all'inizio per il check skip
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.json")

    history: List[Dict[str, float]] = []
    best_miou = -1.0
    best_path = os.path.join(run_dir, "best_checkpoint.pth")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, use_amp=use_amp)
        val_loss, val_metrics = evaluate(model, val_loader, device, num_classes=num_classes, use_amp=use_amp)
        elapsed = time.time() - start_time
        
        val_miou = val_metrics["mIoU"]
        val_dice = val_metrics["Dice"]

        print(
            f"[{dataset_name}] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f} val_Dice={val_dice:.4f} "
            f"time={elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            "time_sec": float(elapsed),
        })
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best mIoU: {best_miou:.4f} (saved)")

    # Test finale con tutte le metriche
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_loss, test_metrics = evaluate(model, test_loader, device, num_classes=num_classes, use_amp=use_amp)
    
    print(f"[{dataset_name}] Test Results:")
    print(f"  mIoU={test_metrics['mIoU']:.4f} Dice={test_metrics['Dice']:.4f}")
    print(f"  Precision={test_metrics['Precision']:.4f} Recall={test_metrics['Recall']:.4f}")
    print(f"  Specificity={test_metrics['Specificity']:.4f} Accuracy={test_metrics['Accuracy']:.4f}")

    final_results = {
        "dataset": dataset_name,
        "model_name": args.model_name,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "best_val_mIoU": float(best_miou),
        "test_loss": float(test_loss),
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
    }
    with open(os.path.join(run_dir, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_model_name(name: str) -> str:
    """Sanitizza il nome del modello per usarlo come nome cartella."""
    name = name.replace("hf_hub:", "")
    name = name.replace("timm/", "")
    name = re.sub(r"[/:\\]", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
    return name


def run_single_model(model_name: str, base_args: argparse.Namespace, datasets: list, all_model_results: Dict[str, List[Dict]]) -> List[Dict]:
    """Esegue training per un singolo modello su tutti i dataset."""
    
    # Usa parametri condivisi da base_args
    embed_dim = getattr(base_args, 'embed_dim', None) or getattr(base_args, 'hidden_dim', 384)
    image_size = base_args.image_size
    
    # Sanitizza nome per output path
    safe_name = sanitize_model_name(model_name)
    
    # Crea args specifici per questo modello
    args = argparse.Namespace(**vars(base_args))
    args.model_name = model_name
    args.hidden_dim = embed_dim
    args.output_dir = os.path.join(base_args.output_dir, safe_name)
    
    # Aggiorna HybridEncoder in_channels con embed_dim corretto
    if hasattr(args, 'HybridEncoder') and args.HybridEncoder:
        args.HybridEncoder = dict(args.HybridEncoder)  # copia per non modificare originale
        args.HybridEncoder['in_channels'] = [embed_dim, embed_dim, embed_dim]
        args.HybridEncoder['hidden_dim'] = embed_dim
        args.HybridEncoder['dim_feedforward'] = embed_dim * 4
    
    print(f"\n{'#'*70}")
    print(f"# MODEL: {model_name}")
    print(f"# embed_dim={embed_dim}, image_size={image_size}")
    print(f"# output_dir={args.output_dir}")
    print(f"{'#'*70}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
    global_csv_path = os.path.join(base_args.output_dir, "all_models_metrics.csv")
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Training {safe_name} on {ds}")
        print(f"{'='*60}")
        result = train_on_dataset(ds, args)
        all_results.append(result)
        
        # Salva CSV incrementale per questo modello
        save_results_csv(all_results, csv_path)
        print(f"[INFO] Updated model CSV: {csv_path} ({len(all_results)}/{len(datasets)} datasets)")
        
        # Aggiorna e salva CSV globale cumulativo
        all_model_results[model_name] = all_results.copy()
        save_global_csv(all_model_results, global_csv_path)
        print(f"[INFO] Updated global CSV: {global_csv_path}")
    
    # Salva risultati aggregati JSON
    with open(os.path.join(args.output_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[DONE] Model {safe_name} completed!")
    return all_results


def save_results_csv(results: List[Dict], csv_path: str) -> None:
    """Salva i risultati in un file CSV."""
    if not results:
        return
    
    # Colonne da includere nel CSV
    columns = [
        "dataset", "model_name", "num_classes", "epochs", "best_val_mIoU",
        "test_loss", "test_mIoU", "test_Dice", "test_Precision", "test_Recall",
        "test_Specificity", "test_Accuracy", "test_F1"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def save_global_csv(all_model_results: Dict[str, List[Dict]], csv_path: str) -> None:
    """Salva tutti i risultati di tutti i modelli in un unico CSV globale."""
    if not all_model_results:
        return
    
    columns = [
        "model_name", "dataset", "num_classes", "epochs", "best_val_mIoU",
        "test_loss", "test_mIoU", "test_Dice", "test_Precision", "test_Recall",
        "test_Specificity", "test_Accuracy", "test_F1"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for model_name, results in all_model_results.items():
            for row in results:
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RadioDino+HybridEncoder segmentation on MedSegBench")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--root", type=str, default="D:/OmniRadio/code/utils")
    parser.add_argument("--model-name", type=str, default="hf_hub:Snarcy/tbd_s")
    parser.add_argument("--image-size", type=int, nargs=2, default=[518, 518])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./outputs/medseg_radio_dinov2_hybrid")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    
    # STA config
    parser.add_argument("--use-sta", type=bool, default=True)
    parser.add_argument("--conv-inplane", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--out-indices", type=int, nargs="+", default=[2, 5, 8, 11])
    
    # HybridEncoder
    parser.add_argument("--use-hybrid-encoder", type=bool, default=True)

    args = parser.parse_args()

    # Load from YAML if provided
    if args.config is not None and os.path.isfile(args.config):
        cfg = load_yaml_config(args.config)
        for k, v in cfg.items():
            k_attr = k.replace("-", "_")
            setattr(args, k_attr, v)

    datasets = args.datasets if args.datasets else getattr(args, "datasets", [
        "BusiMSBench",
        "CovidQUExMSBench", 
        "FHPsAOPMSBench",
        "MosMedPlusMSBench",
        "PandentalMSBench",
        "Promise12MSBench",
        "UltrasoundNerveMSBench",
        "USforKidneyMSBench",
    ])

    # Controlla se c'è una lista di modelli
    models = getattr(args, "models", None)
    
    # Dict per raccogliere tutti i risultati per il CSV globale
    all_model_results: Dict[str, List[Dict]] = {}
    
    if models and isinstance(models, list) and len(models) > 0:
        # Itera su multipli modelli (lista semplice di nomi stringa)
        print(f"\n{'#'*70}")
        print(f"# MULTI-MODEL EXPERIMENT")
        print(f"# Models to test: {len(models)}")
        for i, m in enumerate(models):
            model_name = m if isinstance(m, str) else m.get('name', str(m))
            print(f"#   {i+1}. {model_name}")
        print(f"{'#'*70}\n")
        
        # Crea cartella output padre
        os.makedirs(args.output_dir, exist_ok=True)
        
        for model_entry in models:
            model_name = model_entry if isinstance(model_entry, str) else model_entry.get('name', str(model_entry))
            try:
                results = run_single_model(model_name, args, datasets, all_model_results)
                all_model_results[model_name] = results
            except Exception as e:
                print(f"[ERROR] Model {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Salva CSV globale finale
        global_csv_path = os.path.join(args.output_dir, "all_models_metrics.csv")
        save_global_csv(all_model_results, global_csv_path)
        print(f"\n[INFO] Final global metrics CSV: {global_csv_path}")
        
    else:
        # Singolo modello (backward compatible)
        os.makedirs(args.output_dir, exist_ok=True)
        all_results = []
        csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
        
        for ds in datasets:
            print(f"\n{'='*60}")
            print(f"Training on {ds}")
            print(f"{'='*60}")
            result = train_on_dataset(ds, args)
            all_results.append(result)
            
            # Salva CSV incrementale dopo ogni dataset
            save_results_csv(all_results, csv_path)
            print(f"[INFO] Updated metrics CSV: {csv_path} ({len(all_results)}/{len(datasets)} datasets)")
        
        print(f"\n[INFO] Final metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
