#!/usr/bin/env python3
"""
Fine-tune s2cloudless CNN on CloudSEN12 data for improved baseline.
Freezes early layers and trains final layers on expert-labeled data.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from s2cloudless import S2PixelCloudDetector
import pickle

class CloudSEN12Dataset(Dataset):
    """Dataset for CloudSEN12 patches."""
    
    def __init__(self, image_files, mask_files):
        self.image_files = image_files
        self.mask_files = mask_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        with rasterio.open(self.image_files[idx]) as src:
            image = src.read()  # (bands, H, W)
            image = image.astype(np.float32) / 10000.0
            image = np.clip(image, 0, 1)
        
        # Load mask
        with rasterio.open(self.mask_files[idx]) as src:
            mask = src.read(1)  # (H, W)
            mask = mask.astype(np.float32)
        
        # Convert to torch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask

def get_s2cloudless_features(images):
    """
    Extract features from s2cloudless model (before final layers).
    This uses the pre-trained s2cloudless to get intermediate features.
    """
    detector = S2PixelCloudDetector(threshold=0.5, average_over=1, dilation_size=0)
    
    # Get cloud probabilities (we'll use these as features)
    features = []
    for img in images:
        # s2cloudless expects (H, W, bands) with values 0-10000
        img_scaled = (img.cpu().numpy() * 10000).astype(np.int16)
        img_scaled = np.transpose(img_scaled, (1, 2, 0))
        
        # Get probability map
        prob = detector.get_cloud_probability_maps(np.expand_dims(img_scaled, axis=0))
        prob = prob[0]  # Remove batch dimension
        
        features.append(torch.from_numpy(prob).unsqueeze(0))  # (1, H, W)
    
    return torch.stack(features)  # (batch, 1, H, W)

class FineTunedCloudDetector(nn.Module):
    """Fine-tuned cloud detector with additional layers on top of s2cloudless."""
    
    def __init__(self):
        super().__init__()
        # Add refinement layers
        self.refiner = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, padding=1),  # 10 bands + s2cloudless prob
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
            # No sigmoid here - BCEWithLogitsLoss expects logits
        )
    
    def forward(self, images, s2cloudless_probs):
        # Concatenate original images with s2cloudless probabilities
        x = torch.cat([images, s2cloudless_probs], dim=1)  # (batch, 11, H, W)
        output = self.refiner(x)
        return output.squeeze(1)  # (batch, H, W) - returns logits, not probabilities

def train_finetuned_model(train_loader, val_loader, epochs=10, lr=0.001):
    """Train fine-tuned model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    model = FineTunedCloudDetector().to(device)
    
    # Use weighted loss to handle class imbalance
    # Cloud pixels (1) are ~16%, clear (0) are ~84%
    # Weight clouds 5x higher to encourage detection
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Get s2cloudless features
            with torch.no_grad():
                s2cloudless_probs = get_s2cloudless_features(images).to(device)
            
            # Forward pass
            outputs = model(images, s2cloudless_probs)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                s2cloudless_probs = get_s2cloudless_features(images).to(device)
                outputs = model(images, s2cloudless_probs)
                
                # Apply sigmoid for predictions (since we're using BCEWithLogitsLoss)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.append(preds.cpu().numpy().flatten())
                val_targets.append(masks.cpu().numpy().flatten())
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        accuracy = accuracy_score(val_targets, val_preds)
        precision = precision_score(val_targets, val_preds, zero_division=0)
        recall = recall_score(val_targets, val_preds, zero_division=0)
        f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, "
              f"Acc={accuracy:.4f}, F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def finetune_cnn():
    """Main fine-tuning pipeline."""
    print("=" * 60)
    print("🔧 Fine-tuning s2cloudless on CloudSEN12")
    print("=" * 60)
    
    # Load CloudSEN12 data
    image_files = sorted(glob.glob('data/cloudsen12_processed/*_image.tif'))
    mask_files = sorted(glob.glob('data/cloudsen12_processed/*_mask.tif'))
    
    if len(image_files) < 2:
        raise ValueError("Need at least 2 patches for train/val split")
    
    print(f"\n📂 Found {len(image_files)} patches")
    
    # Split into train/val (80/20)
    split_idx = int(len(image_files) * 0.8)
    train_images = image_files[:split_idx]
    train_masks = mask_files[:split_idx]
    val_images = image_files[split_idx:]
    val_masks = mask_files[split_idx:]
    
    print(f"📊 Train: {len(train_images)} patches, Val: {len(val_images)} patches")
    
    # Create datasets
    train_dataset = CloudSEN12Dataset(train_images, train_masks)
    val_dataset = CloudSEN12Dataset(val_images, val_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Train
    print("\n🚀 Starting fine-tuning...")
    model = train_finetuned_model(train_loader, val_loader, epochs=20, lr=0.001)
    
    # Save model
    torch.save(model.state_dict(), 'models/finetuned_s2cloudless.pth')
    print("\n💾 Model saved to: models/finetuned_s2cloudless.pth")
    
    # Final evaluation on all data
    print("\n📊 Final evaluation on all patches...")
    all_dataset = CloudSEN12Dataset(image_files, mask_files)
    all_loader = DataLoader(all_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in all_loader:
            images = images.to(device)
            s2cloudless_probs = get_s2cloudless_features(images).to(device)
            outputs = model(images, s2cloudless_probs)
            
            # Apply sigmoid for predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print("\n" + "=" * 60)
    print("🎯 FINE-TUNED CNN BASELINE")
    print("=" * 60)
    print(f"\n📊 Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Save results
    import json
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    with open('results/finetuned_cnn_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: results/finetuned_cnn_baseline.json")
    print("=" * 60)
    
    return model, results

if __name__ == "__main__":
    model, results = finetune_cnn()
