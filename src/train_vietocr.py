import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from vietocr.model.trainer import Trainer
from vietocr.model.vocab import Vocab
from vietocr.model.seqmodel import SeqModel
from vietocr.model.transformer import LanguageTransformer
from transformers import ViTModel, ViTConfig
import torch.nn as nn

class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.projection = nn.Linear(768, config['transformer']['d_model'])
        
    def forward(self, x):
        outputs = self.vit(x)
        hidden_states = outputs.last_hidden_state
        return self.projection(hidden_states)

class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_file, vocab, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.vocab = vocab
        
        # Load labels
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # Create list of image files
        self.image_files = list(self.labels.keys())
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = self.labels[img_name]
        
        # Convert label to indices
        label_indices = self.vocab.encode(label)
        
        return {
            'img': image,
            'label': label_indices,
            'text': label
        }

def create_vocab(labels_file):
    # Create vocabulary from all labels
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # Get all unique characters
    chars = set()
    for label in labels.values():
        chars.update(label)
    
    # Create vocab
    vocab = Vocab(chars)
    return vocab

def create_vit_config():
    config = {
        'transformer': {
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'activation': 'relu',
            'normalize_before': False,
            'return_intermediate_dec': False
        },
        'train': {
            'batch_size': 8,
            'epochs': 100,
            'print_every': 200,
            'valid_every': 1000,
            'checkpoint': './checkpoint/vit_transformer_checkpoint.pth',
            'export': './weights/vit_transformer.pth',
            'max_iters': 10000,
            'dropout': 0.1,
            'learning_rate': 1e-4
        }
    }
    return config

def train():
    # Configuration
    config = create_vit_config()
    
    # Create vocabulary
    vocab = create_vocab('dataset/labels.json')
    
    # Create dataset
    train_dataset = CustomDataset(
        data_dir='dataset/data',
        labels_file='dataset/labels.json',
        vocab=vocab
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # Create model
    encoder = ViTEncoder(config)
    transformer = LanguageTransformer(config['transformer'])
    model = SeqModel(encoder, transformer, vocab)
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Train model
    trainer.train(train_loader)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs('./weights', exist_ok=True)
    
    # Start training
    train() 