import argparse
import torch
import torch.nn as nn
from torchvision.transforms import v2
import csv
import os
import numpy as np
#core functions
from image_dataset import get_image_dataset
from classification_utils import train_loop, eval_loop,get_metrics
from torch.utils.data import default_collate
import timm

#arguments
parser = argparse.ArgumentParser(description='Fine-tune image classification models on medical datasets')
parser.add_argument('--model', default='hf_hub:Snarcy/RadioDino-s16', type=str, help='Model name')
parser.add_argument('--dataset_path_train', type=str, help='Path to the dataset for training')
parser.add_argument('--dataset_path_val', type=str, help='Path to the dataset for validation')
parser.add_argument('--dataset_path_test', type=str, help='Path to the dataset for testing')
parser.add_argument('--output_path', type=str, help='Path to the output folder')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
parser.add_argument('--device', default='cuda', type=str, help='Device')
parser.add_argument('--epochs', default=40, type=int, help='Number of epochs')
parser.add_argument('--warmup_epochs', default=10, type=int, help='Number of warmup epochs')
parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
parser.add_argument('--dataset_name', type=str, help='Dataset')
parser.add_argument('--use_amp', default=True, type=bool, help='Use automatic mixed precision')
parser.add_argument('--gradient_clip', default=1, type=float, help='Gradient clipping')
parser.add_argument('--patience', default=10000, type=int, help='Patience for early stopping')
parser.add_argument('--cutmix', default=False, type=bool, help='Use cutmix')
parser.add_argument('--mixup', default=False, type=bool, help='Use mixup')
parser.add_argument('--seed', default=-1, type=int, help='Seed')
parser.add_argument('--dropout', default=0.4, type=float, help='Model dropout')
#log and result files
parser.add_argument('--log_file', default='log.csv', type=str, help='Log file name')
parser.add_argument('--result_file', default='results_classification.csv', type=str, help='Result file name')

#adapters args for LoRA
parser.add_argument('--lora_r', default=16, type=int, help='LoRA rank')
parser.add_argument('--lora_alpha', default=32, type=int, help='LoRA alpha')
parser.add_argument('--lora_dropout', default=0.1, type=float, help='LoRA dropout')
parser.add_argument('--use_lora', action='store_true')

args = parser.parse_args()

#set seed
if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

#model creation
def create_model(num_classes, model_name):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, drop_rate=args.dropout)
    if args.use_lora:
        print("Using LoRA adapters")
        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules='all-linear',
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, lora_config)
    else:
        print("Full tuning")
    
    #print trainable parameters and total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params} / Total parameters: {total_params} ({100 * trainable_params / total_params:.2f}%)')
    model.to(args.device)
    return model

def create_dataset(path, is_train, no_workers=False):
    # default transform for dino models
    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)
    if is_train:
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((224,224)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(7),
            v2.RandomResizedCrop(224, scale=(0.85,1.0)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(DEFAULT_MEAN, DEFAULT_STD)
        ])

    else:
        transform = v2.Compose([
            v2.ToImage(), 
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    ])
    #args for dataset
    dataset_args = argparse.Namespace(
        data_dir=path,
        batch_size=args.batch_size,
        num_workers=args.num_workers if not no_workers else 0,
        transform=transform,
        persistent_workers=True if args.num_workers > 0 and not no_workers else False,
        shuffle=True if is_train else False)
    #get dataset
    dataset = get_image_dataset(dataset_args)
    return dataset
if __name__ == '__main__':
    #print args info in a good format
    print("Arguments:")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    #create output path
    #create dataset
    train_loader = create_dataset(args.dataset_path_train, is_train=True)
    val_loader = create_dataset(args.dataset_path_val, is_train=False)
    test_loader = create_dataset(args.dataset_path_test, is_train=False, no_workers=True)
    #get classes
    classes = train_loader.dataset.classes
  
    print("Classes:", classes)
    #log files
    log_file = args.log_file
    result_file = args.result_file
    #create model
    model = create_model(len(classes), args.model)
    #sanitize model name for file paths and remove hf_hub:Snarcy/ prefix
    args.model = args.model.replace('/', '_').replace(':', '_').replace('hf_hub_Snarcy_', '')
    
    args.output_path = os.path.join(args.output_path, args.dataset_name)
    #if the output path and model folder do not exist, skip the training
    
    if os.path.exists(os.path.join(args.output_path, args.model, 'best.pth')):
        print(f'Model {args.model} already trained on dataset {args.dataset_name}, skipping training.')
        exit(0)
    log_file = os.path.join(args.output_path, args.model, log_file)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    from collections import Counter

    train_labels = train_loader.dataset.targets
    freq = Counter(train_labels)
    num_classes = len(classes)

    weights = torch.tensor(
        [ (len(train_labels) / freq[i])**0.5 for i in range(num_classes) ],
        dtype=torch.float32
    ).to(args.device)
    print("Class weights:", weights)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # --- Scheduler with proper warmup ---
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )

    best_val_f1 = 0.0
    patience = args.patience
    last_patience_losses = []

    for epoch in range(args.epochs):

        # stronger clipping during warmup
        current_clip = 0.5 if epoch < args.warmup_epochs else args.gradient_clip

        train_loss = train_loop(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device,
            use_amp=args.use_amp,
            gradient_clip=current_clip
        )

        val_loss, logits, targets = eval_loop(
            model,
            val_loader,
            criterion,
            args.device
        )

        accuracy, precision, recall, f1, auc = get_metrics(logits, targets)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"AUC: {auc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if f1 >= best_val_f1:
            best_val_f1 = f1
            os.makedirs(f"{args.output_path}/{args.model}", exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{args.output_path}/{args.model}/best.pth"
            )

        # logging stays identical
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, args.model, accuracy, precision, recall, f1, auc])

        # early stopping logic unchanged
        last_patience_losses.append(val_loss)
        if len(last_patience_losses) > patience:
            last_patience_losses.pop(0)
            if val_loss > min(last_patience_losses):
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # test on the test set
    model.load_state_dict(torch.load(f'{args.output_path}/{args.model}/best.pth'))
    test_loss, logits, targets = eval_loop(model, test_loader, criterion, args.device)
    accuracy, precision, recall, f1, auc = get_metrics(logits, targets)
    print(f'Test Loss: {test_loss} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1} | AUC: {auc}')
    # Log the results
    if not os.path.exists(result_file):
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset','Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if args.use_lora:
            writer.writerow([args.dataset_name, args.model + '_lora', accuracy, precision, recall, f1, auc])
        else:
            writer.writerow([args.dataset_name, args.model, accuracy, precision, recall, f1, auc])