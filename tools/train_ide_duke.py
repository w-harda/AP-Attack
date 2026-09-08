#!/usr/bin/env python3
"""Train a DukeMTMC-reID IDE surrogate compatible with AP-Attack Stage 2."""

import argparse
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.nn import init
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 702
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FILENAME_PATTERN = re.compile(r"([\-\d]+)_c(\d)")


class IDE(nn.Module):
    """State-dict compatible copy of the IDE used by train_ap_attack.py."""

    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=1024, norm=False, dropout=0, num_classes=0):
        super(IDE, self).__init__()

        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.base = torchvision.models.resnet50(pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode="fan_out")
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, is_training=False, metaTrain=True, mix_thre=0.6,
                mix_pro=0.5, output_both=False, mix_info=None, lamd=None):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x_layer0 = self.base.maxpool(x)
        x_layer1 = self.base.layer1(x_layer0)
        x_layer2 = self.base.layer2(x_layer1)
        x_layer3 = self.base.layer3(x_layer2)
        feat_map = self.base.layer4(x_layer3)
        x = feat_map

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)

        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            logits = self.classifier(x)

        if is_training:
            return logits, x
        return x

    def reset_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for key in param_dict:
            self.state_dict()[key.replace("module.", "")].copy_(param_dict[key])


class RandomSizedRectCrop:
    """Classic IDE/CamStyle random rectangle crop with a resize fallback."""

    def __init__(self, height, width, target_area=(0.64, 1.0), aspect_ratio=(2.0, 3.0)):
        self.height = height
        self.width = width
        self.target_area = target_area
        self.aspect_ratio = aspect_ratio

    def __call__(self, image):
        image_width, image_height = image.size
        area = image_width * image_height

        for _ in range(10):
            target_area = random.uniform(*self.target_area) * area
            aspect_ratio = random.uniform(*self.aspect_ratio)
            crop_height = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_width = int(round(math.sqrt(target_area / aspect_ratio)))

            if crop_width <= image_width and crop_height <= image_height:
                x1 = random.randint(0, image_width - crop_width)
                y1 = random.randint(0, image_height - crop_height)
                image = image.crop((x1, y1, x1 + crop_width, y1 + crop_height))
                return image.resize((self.width, self.height), Image.BILINEAR)

        return image.resize((self.width, self.height), Image.BILINEAR)


class DukeImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, pid, camid = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), pid, camid


def parse_duke_split(directory, relabel=False):
    image_paths = sorted(Path(directory).glob("*.jpg"))
    samples = []
    identities = []
    for image_path in image_paths:
        match = FILENAME_PATTERN.search(image_path.name)
        if match is None:
            raise RuntimeError("Unexpected DukeMTMC-reID filename: {}".format(image_path))
        pid, camid = map(int, match.groups())
        if pid == -1:
            continue
        identities.append(pid)
        samples.append((image_path, pid, camid - 1))

    if relabel:
        pid_to_label = {pid: label for label, pid in enumerate(sorted(set(identities)))}
        samples = [(image_path, pid_to_label[pid], camid) for image_path, pid, camid in samples]

    return samples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return worker_init_fn


def build_dataloaders(data_root, batch_size, workers, seed):
    data_root = Path(data_root)
    train_samples = parse_duke_split(data_root / "bounding_box_train", relabel=True)
    query_samples = parse_duke_split(data_root / "query")
    gallery_samples = parse_duke_split(data_root / "bounding_box_test")

    train_pids = {pid for _, pid, _ in train_samples}
    if len(train_pids) != NUM_CLASSES:
        raise RuntimeError("Expected {} train identities, found {}".format(NUM_CLASSES, len(train_pids)))

    train_transform = T.Compose([
        RandomSizedRectCrop(IMAGE_HEIGHT, IMAGE_WIDTH),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = T.Compose([
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "worker_init_fn": make_worker_init_fn(seed),
    }
    train_loader = DataLoader(
        DukeImageDataset(train_samples, train_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs
    )
    query_loader = DataLoader(
        DukeImageDataset(query_samples, test_transform),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )
    gallery_loader = DataLoader(
        DukeImageDataset(gallery_samples, test_transform),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    print(
        "DukeMTMC-reID: train {} IDs / {} images, query {} IDs / {} images, "
        "gallery {} IDs / {} images".format(
            len(train_pids), len(train_samples),
            len({pid for _, pid, _ in query_samples}), len(query_samples),
            len({pid for _, pid, _ in gallery_samples}), len(gallery_samples),
        )
    )
    return train_loader, query_loader, gallery_loader


def adjust_learning_rate(optimizer, epoch):
    factor = 0.1 if epoch >= 40 else 1.0
    optimizer.param_groups[0]["lr"] = 0.01 * factor
    optimizer.param_groups[1]["lr"] = 0.1 * factor


def extract_features(model, data_loader, device):
    features, pids, camids = [], [], []
    model.eval()
    with torch.no_grad():
        for images, batch_pids, batch_camids in data_loader:
            embeddings = model(images.to(device))
            features.append(embeddings.cpu())
            pids.append(batch_pids)
            camids.append(batch_camids)
    return torch.cat(features), torch.cat(pids).numpy(), torch.cat(camids).numpy()


def evaluate_duke(model, query_loader, gallery_loader, device):
    query_features, query_pids, query_camids = extract_features(model, query_loader, device)
    gallery_features, gallery_pids, gallery_camids = extract_features(model, gallery_loader, device)
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)
    distance = (
        query_features.pow(2).sum(dim=1, keepdim=True)
        + gallery_features.pow(2).sum(dim=1).unsqueeze(0)
        - 2.0 * query_features.mm(gallery_features.t())
    ).numpy()

    indices = np.argsort(distance, axis=1)
    all_cmc, all_ap = [], []
    for query_index, order in enumerate(indices):
        query_pid = query_pids[query_index]
        query_camid = query_camids[query_index]
        keep = ~((gallery_pids[order] == query_pid) & (gallery_camids[order] == query_camid))
        matches = (gallery_pids[order][keep] == query_pid).astype(np.int32)
        if not np.any(matches):
            continue

        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)
        precision = matches.cumsum() / (np.arange(len(matches)) + 1.0)
        all_ap.append(float((precision * matches).sum() / matches.sum()))

    if not all_cmc:
        raise RuntimeError("No valid Duke query has a matching gallery image")

    cmc = np.asarray([curve[:10] for curve in all_cmc]).mean(axis=0)
    return {
        "mAP": float(np.mean(all_ap)),
        "Rank-1": float(cmc[0]),
        "Rank-5": float(cmc[4]),
        "Rank-10": float(cmc[9]),
    }


def checkpoint_compatibility_test(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = IDE(num_classes=NUM_CLASSES, pretrained=False)
    expected_state_dict = model.state_dict()
    missing_keys = sorted(set(expected_state_dict) - set(checkpoint))
    unexpected_keys = sorted(set(checkpoint) - set(expected_state_dict))
    shape_mismatches = [
        key for key, value in checkpoint.items()
        if key in expected_state_dict and value.shape != expected_state_dict[key].shape
    ]
    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            "Checkpoint compatibility failed: missing_keys={}, unexpected_keys={}, shape_mismatches={}".format(
                missing_keys, unexpected_keys, shape_mismatches
            )
        )
    incompatible_keys = model.load_state_dict(checkpoint, strict=True)
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        raise RuntimeError("Checkpoint compatibility failed after strict load")
    print("checkpoint compatibility test OK")


def save_training_checkpoint(path, epoch, model, optimizer, metrics, best_mAP):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "mAP": metrics["mAP"],
        "Rank-1": metrics["Rank-1"],
        "best_mAP": best_mAP,
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DukeMTMC-reID IDE surrogate for AP-Attack")
    parser.add_argument("--data-root", default="/home/lzf/ldx/datasets/DukeMTMC-reID")
    parser.add_argument("--output-dir", default="/home/lzf/ldx/checkpoints/AP-Attack/ide")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-period", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", default="", help="Path to ide_dukemtmcreid_training.pth.tar")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for IDE surrogate training")

    set_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, query_loader, gallery_loader = build_dataloaders(
        args.data_root, args.batch_size, args.workers, args.seed
    )

    model = IDE(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        [
            {"params": model.base.parameters(), "lr": 0.01},
            {
                "params": list(model.feat.parameters())
                + list(model.feat_bn.parameters())
                + list(model.classifier.parameters()),
                "lr": 0.1,
            },
        ],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch, best_mAP = 1, -float("inf")
    latest_metrics = {"mAP": 0.0, "Rank-1": 0.0, "Rank-5": 0.0, "Rank-10": 0.0}

    if args.resume:
        resume_state = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_state["state_dict"])
        optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = resume_state["epoch"] + 1
        best_mAP = resume_state.get("best_mAP", resume_state["mAP"])
        latest_metrics["mAP"] = resume_state["mAP"]
        latest_metrics["Rank-1"] = resume_state["Rank-1"]
        print("Resumed training from epoch {}".format(resume_state["epoch"]))

    best_path = output_dir / "ide_dukemtmcreid_best.pth"
    last_path = output_dir / "ide_dukemtmcreid_last.pth"
    training_path = output_dir / "ide_dukemtmcreid_training.pth.tar"

    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        model.train()
        total_loss, total_images = 0.0, 0

        for images, pids, _ in train_loader:
            images = images.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, _ = model(images, is_training=True)
            loss = criterion(logits, pids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)

        print(
            "Epoch {}/{}: loss {:.4f}, backbone_lr {:.5f}, head_lr {:.5f}".format(
                epoch, args.epochs, total_loss / total_images,
                optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
            )
        )

        if epoch % args.eval_period == 0 or epoch == args.epochs:
            latest_metrics = evaluate_duke(model, query_loader, gallery_loader, device)
            print(
                "Validation epoch {}: Rank-1 {:.2%}, Rank-5 {:.2%}, Rank-10 {:.2%}, mAP {:.2%}".format(
                    epoch,
                    latest_metrics["Rank-1"],
                    latest_metrics["Rank-5"],
                    latest_metrics["Rank-10"],
                    latest_metrics["mAP"],
                )
            )
            if latest_metrics["mAP"] > best_mAP:
                best_mAP = latest_metrics["mAP"]
                torch.save(model.state_dict(), best_path)
                print("Saved best checkpoint: {}".format(best_path))

        torch.save(model.state_dict(), last_path)
        save_training_checkpoint(training_path, epoch, model, optimizer, latest_metrics, best_mAP)

    checkpoint_compatibility_test(best_path)


if __name__ == "__main__":
    main()
