"""
RadioDino con HybridEncoder - Versione potenziata del RadioDinoSegModel.

Aggiunge HybridEncoder tra backbone e decoder per migliorare le performance,
come fa DEIMv2 con DINOv3STAs.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.deim.segmentation import UNetSegmentationHead
from engine.deim.radio_dino_seg_model import RadioDinoSTAs
from engine.deim.hybrid_encoder import HybridEncoder


class OmniRadHybridSegModel(nn.Module):
	"""Modello completo per segmentazione con RadioDino + HybridEncoder + U-Net head.
	
	Architettura (come DEIMv2):
	1. RadioDinoSTAs: backbone ViT + CNN parallela per feature multi-scala
	2. HybridEncoder: feature pyramid network con CSP blocks e transformer
	3. UNetSegmentationHead: decoder U-Net-like per produrre le mask
	
	Questo design usa lo stesso HybridEncoder di DEIMv2 per colmare il gap
	di performance rispetto a DINOv3STAs.
	"""

	def __init__(
		self,
		model_name: str,
		num_classes: int,
		image_size: Tuple[int, int],
		hidden_dim: Optional[int] = None,
		use_sta: bool = True,
		conv_inplane: int = 16,
		out_indices: Tuple[int, ...] = (2, 5, 8, 11),
		use_sync_bn: bool = True,  # Default True come DINOv3STAs
		# HybridEncoder
		use_hybrid_encoder: bool = True,
		encoder_cfg: Optional[Dict] = None,  # Sezione HybridEncoder dal YAML
		# Freeze backbone
        freeze_backbone: bool = False,
	) -> None:
		super().__init__()

		self.image_size = image_size
		self.use_hybrid_encoder = use_hybrid_encoder
		self.freeze_backbone = freeze_backbone  # Salva il flag per usarlo in forward
		
		# Backbone con STA
		self.backbone = RadioDinoSTAs(
			model_name=model_name,
			image_size=image_size,
			hidden_dim=hidden_dim,
			use_sta=use_sta,
			conv_inplane=conv_inplane,
			out_indices=out_indices,
			pretrained=True,
			use_sync_bn=use_sync_bn,
		)
		
		backbone_channels = self.backbone.out_channels  # [hidden_dim, hidden_dim, hidden_dim]
		
		if use_hybrid_encoder:
			# Default config (come DEIMv2 DINOv3STAs)
			cfg = {
				'in_channels': backbone_channels,
				'feat_strides': [8, 16, 32],
				'hidden_dim': 256,
				'dim_feedforward': 1024,
				'expansion': 1.0,
				'depth_mult': 1.0,
				'use_encoder_idx': [2],
				'num_encoder_layers': 1,
				'nhead': 8,
				'dropout': 0.0,
				'enc_act': 'gelu',
				'act': 'silu',
				'version': 'deim',
				'csp_type': 'csp2',
				'fuse_op': 'sum',
			}
			# Override con encoder_cfg se fornito (tutti i parametri configurabili)
			if encoder_cfg is not None:
				cfg.update(encoder_cfg)
			
			self.encoder = HybridEncoder(
				in_channels=cfg.get('in_channels', backbone_channels),
				feat_strides=cfg['feat_strides'],
				hidden_dim=cfg['hidden_dim'],
				dim_feedforward=cfg['dim_feedforward'],
				expansion=cfg['expansion'],
				depth_mult=cfg['depth_mult'],
				use_encoder_idx=cfg['use_encoder_idx'],
				num_encoder_layers=cfg['num_encoder_layers'],
				nhead=cfg['nhead'],
				dropout=cfg['dropout'],
				enc_act=cfg['enc_act'],
				act=cfg['act'],
				version=cfg['version'],
				csp_type=cfg['csp_type'],
				fuse_op=cfg['fuse_op'],
			)
			
			in_channels = list(self.encoder.out_channels)
			_hidden_dim = cfg['hidden_dim']
			print(f"[RadioDinoHybridSegModel] HybridEncoder: hidden_dim={cfg['hidden_dim']}, expansion={cfg['expansion']}, depth_mult={cfg['depth_mult']}")
		else:
			self.encoder = None
			in_channels = backbone_channels
			_hidden_dim = self.backbone.hidden_dim

		# Testa di segmentazione U-Net
		self.seg_head = UNetSegmentationHead(
			in_channels=in_channels,
			num_classes=num_classes,
			hidden_dim=_hidden_dim,
			act="relu",
			final_upsample=True,
			eval_spatial_size=image_size,
		)

	def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		B, C, H, W = x.shape
		if self.freeze_backbone:
			with torch.no_grad():
				c2, c3, c4 = self.backbone(x)
		else:
			c2, c3, c4 = self.backbone(x)
		feats = [c2, c3, c4]
		
		# Passa attraverso HybridEncoder se presente
		if self.use_hybrid_encoder and self.encoder is not None:
			# HybridEncoder richiede dimensioni compatibili per upsampling/downsampling
			# Forza le dimensioni a essere esattamente H/8, H/16, H/32
			target_sizes = [(H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)]
			aligned_feats = []
			for feat, target_size in zip(feats, target_sizes):
				if feat.shape[-2:] != target_size:
					feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
				aligned_feats.append(feat)
			feats = self.encoder(aligned_feats)
		
		# Passa al decoder
		out = self.seg_head(feats, targets=None)
		return out
