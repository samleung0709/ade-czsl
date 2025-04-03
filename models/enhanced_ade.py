import torch
import torch.nn as nn
import torch.nn.functional as F
from .ade import ADE
from CLIP import clip
from .multi_head_attention import CrossAttention
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class CLIPCompositionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = CrossAttention(dim, num_heads=8)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(p=0.1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-Attention
        attn_out, attn_weights = self.self_attn(q=x, k=x, return_attention=True)
        x = self.norm1(x + attn_out)
        
        # MLP
        x = self.norm2(x + self.mlp(x))
        return x, attn_weights

class EnhancedADE(ADE):
    def __init__(self, dset, args):
        super().__init__(dset, args)
        
        # Initialize CLIP model and preprocessing
        self.clip_model, _ = clip.load(args.clip_visual, device='cuda')
        self.clip_preprocess = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Feature dimensions
        self.clip_dim = 512
        self.ade_dim = self.args.emb_dim
        
        # Projections and modules
        self.clip_proj = nn.Linear(self.clip_dim, self.ade_dim)
        self.clip_composer = CLIPCompositionModule(self.ade_dim)
        self.afm = nn.Sequential(
            nn.LayerNorm(self.ade_dim * 2),
            nn.Linear(self.ade_dim * 2, self.ade_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.ade_dim, self.ade_dim)
        )

    def extract_clip_features(self, images):
        """Extract CLIP features from raw images"""
        # Convert images to CLIP format (B, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Ensure images are in correct format
        images = images.permute(0, 3, 1, 2) if images.shape[1] != 3 else images
        
        # Preprocess for CLIP
        preprocessed = []
        for img in images:
            preprocessed.append(self.clip_preprocess(img.cpu()).unsqueeze(0))
        preprocessed = torch.cat(preprocessed).to(images.device)
        
        # Extract features
        with torch.no_grad():
            features = self.clip_model.encode_image(preprocessed)
        return features

    def forward(self, x):
        # Original ADE forward pass for training data
        ade_loss, ade_pred, ade_scores = super().forward(x)
        
        # Get raw images
        if self.training:
            images = x[0]  # During training
        else:
            images = x     # During testing
            
        # Extract CLIP features
        clip_features = self.extract_clip_features(images)
        clip_features = self.clip_proj(clip_features)
        
        # CLIP compositional features
        clip_comp_features, _ = self.clip_composer(clip_features.unsqueeze(1))
        clip_comp_features = clip_comp_features.squeeze(1)
        
        # Get ADE features
        ade_features = self.image_embedder(
            self.cross_attn(q=images, k=images)[0][:,0,:]
        )
        
        # Adaptive fusion
        fused_features = self.afm(
            torch.cat([ade_features, clip_comp_features], dim=1)
        )
        
        if self.training:
            concept = self.compose_word_embeddings()
            pair_pred = self.cos_logits(fused_features, concept)
            return ade_loss, pair_pred, ade_scores
        else:
            return fused_features