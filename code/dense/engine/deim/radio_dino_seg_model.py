from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.deim.segmentation import UNetSegmentationHead


class SpatialPriorModule(nn.Module):
	"""Modulo CNN parallelo per estrarre feature spaziali locali (stile DEIMv2).
	
	Produce 3 livelli di feature a scale 1/8, 1/16, 1/32 che vengono
	fuse con le feature semantiche del ViT per migliorare i dettagli locali.
	"""
	
	def __init__(self, inplanes: int = 16, use_sync_bn: bool = True) -> None:
		super().__init__()
		
		# DINOv3STAs usa sempre SyncBatchNorm
		norm_layer = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
		
		# 1/4 - stem
		self.stem = nn.Sequential(
			nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
			norm_layer(inplanes),
			nn.GELU(),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)
		# 1/8
		self.conv2 = nn.Sequential(
			nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
			norm_layer(2 * inplanes),
		)
		# 1/16
		self.conv3 = nn.Sequential(
			nn.GELU(),
			nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
			norm_layer(4 * inplanes),
		)
		# 1/32
		self.conv4 = nn.Sequential(
			nn.GELU(),
			nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
			norm_layer(4 * inplanes),
		)
		
		# Canali di output per ogni scala
		self.out_channels = [2 * inplanes, 4 * inplanes, 4 * inplanes]
	
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		c1 = self.stem(x)       # 1/4
		c2 = self.conv2(c1)     # 1/8
		c3 = self.conv3(c2)     # 1/16
		c4 = self.conv4(c3)     # 1/32
		return c2, c3, c4


class RadioDinoSTAs(nn.Module):
	"""Backbone RadioDino/DINOv2 (timm) con Spatial-Token Attention (STA).
	
	Architettura ispirata a DINOv3STAs di DEIMv2:
	- Estrae feature multi-scala dal ViT (layer intermedi)
	- CNN parallela (SpatialPriorModule) per catturare dettagli locali
	- Fusione delle feature semantiche ViT + feature spaziali CNN
	- Proiezione finale con conv 1x1 + norm
	
	Questo approccio è superiore per la segmentazione medica dove
	i dettagli locali (bordi, texture) sono fondamentali.
	"""
	
	def __init__(
		self,
		model_name: str,
		image_size: Tuple[int, int],
		patch_size: int = 14,
		hidden_dim: Optional[int] = None,
		use_sta: bool = True,
		conv_inplane: int = 16,
		out_indices: Tuple[int, ...] = (2, 5, 8, 11),  # Layer intermedi da estrarre
		pretrained: bool = True,
		use_sync_bn: bool = False,
	) -> None:
		super().__init__()
		import timm
		
		self.image_size = image_size
		self.patch_size = patch_size
		self.use_sta = use_sta
		self.out_indices = out_indices
		
		# Crea il backbone ViT con supporto per dimensioni dinamiche
		# NON usiamo features_only, vogliamo accesso ai layer intermedi
		try:
			self.vit = timm.create_model(
				model_name,
				pretrained=pretrained,
				img_size=image_size,
				dynamic_img_size=True,
				num_classes=0,  # Rimuove la testa di classificazione
			)
		except TypeError:
			print(f"[WARN] {model_name} non supporta dynamic_img_size")
			self.vit = timm.create_model(
				model_name,
				pretrained=pretrained,
				img_size=image_size,
				num_classes=0,
			)
		
		# Determina embed_dim dal modello
		self.embed_dim = getattr(self.vit, 'embed_dim', 384)
		if hasattr(self.vit, 'num_features'):
			self.embed_dim = self.vit.num_features
		
		# Determina patch_size dal modello se possibile
		if hasattr(self.vit, 'patch_embed') and hasattr(self.vit.patch_embed, 'patch_size'):
			ps = self.vit.patch_embed.patch_size
			self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
		
		# SpatialPriorModule per feature locali
		if use_sta:
			self.sta = SpatialPriorModule(inplanes=conv_inplane, use_sync_bn=use_sync_bn)
			sta_channels = self.sta.out_channels
			print(f"[RadioDinoSTAs] Using STA with inplanes={conv_inplane}, channels={sta_channels}")
		else:
			self.sta = None
			sta_channels = [0, 0, 0]
			conv_inplane = 0
		
		# Dimensione hidden per la proiezione finale
		_hidden_dim = hidden_dim if hidden_dim is not None else self.embed_dim
		self.hidden_dim = _hidden_dim
		
		# Conv 1x1 per fondere e proiettare le feature (come DINOv3STAs)
		# 3 scale: 1/8, 1/16, 1/32
		self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, _hidden_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, _hidden_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, _hidden_dim, 1, bias=False),
        ])

		
		# Usa SyncBatchNorm come DINOv3STAs
		self.norms = nn.ModuleList([
			nn.SyncBatchNorm(_hidden_dim),
			nn.SyncBatchNorm(_hidden_dim),
			nn.SyncBatchNorm(_hidden_dim),
		])
		
		# Output channels per l'interfaccia con UNetSegmentationHead
		self.out_channels = [_hidden_dim, _hidden_dim, _hidden_dim]
		
		print(f"[RadioDinoSTAs] model={model_name}")
		print(f"  embed_dim={self.embed_dim}, patch_size={self.patch_size}")
		print(f"  image_size={image_size}, hidden_dim={_hidden_dim}")
		print(f"  out_indices={out_indices}, use_sta={use_sta}")
	
	def _extract_intermediate_features(self, x: torch.Tensor) -> List[torch.Tensor]:
		"""Estrae feature dai layer intermedi del ViT."""
		B, C, H, W = x.shape
		
		# Calcola dimensioni della griglia di patch
		H_patches = H // self.patch_size
		W_patches = W // self.patch_size
		
		
		# Forward manuale attraverso i blocchi
		if not hasattr(self.vit, 'patch_embed'):
			raise ValueError("Modello ViT non supportato: manca patch_embed")
		
		x_embed = self.vit.patch_embed(x)
		
		# Gestisci output patch_embed: [B, H_p, W_p, D] -> [B, N, D]
		if x_embed.dim() == 4:
			B_e, d1, d2, d3 = x_embed.shape
			if d3 == self.embed_dim:
				# [B, H_p, W_p, D] -> [B, N, D]
				x_embed = x_embed.view(B_e, -1, self.embed_dim)
			elif d1 == self.embed_dim:
				# [B, D, H_p, W_p] -> [B, N, D]
				x_embed = x_embed.flatten(2).transpose(1, 2)
			else:
				raise ValueError(f"patch_embed shape inattesa: {x_embed.shape}")
		
		# Aggiungi cls token se presente
		num_prefix_tokens = 0
		if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
			cls_tokens = self.vit.cls_token.expand(x_embed.shape[0], -1, -1)
			x_embed = torch.cat([cls_tokens, x_embed], dim=1)
			num_prefix_tokens += 1
		
		# Aggiungi position embedding
		if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
			pos_embed = self.vit.pos_embed
			if pos_embed.shape[1] != x_embed.shape[1]:
				# Interpola pos_embed
				pos_prefix = pos_embed[:, :num_prefix_tokens, :] if num_prefix_tokens > 0 else None
				pos_patches = pos_embed[:, num_prefix_tokens:, :]
				orig_size = int(pos_patches.shape[1] ** 0.5)
				D = pos_patches.shape[-1]
				pos_patches = pos_patches.reshape(1, orig_size, orig_size, D).permute(0, 3, 1, 2)
				pos_patches = F.interpolate(pos_patches, size=(H_patches, W_patches), mode='bicubic', align_corners=False)
				pos_patches = pos_patches.permute(0, 2, 3, 1).reshape(1, -1, D)
				pos_embed = torch.cat([pos_prefix, pos_patches], dim=1) if pos_prefix is not None else pos_patches
			x_embed = x_embed + pos_embed
		
		# Passa attraverso i blocchi
		blocks = self.vit.blocks if hasattr(self.vit, 'blocks') else self.vit.layers
		
		feats = []
		for i, blk in enumerate(blocks):
			x_embed = blk(x_embed)
			if i in self.out_indices:
				feat = x_embed[:, num_prefix_tokens:, :]
				feat = feat.transpose(1, 2).view(B, -1, H_patches, W_patches).contiguous()
				feats.append(feat)
		
		return feats
	
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Forward pass che produce 3 livelli di feature multi-scala.
		
		Returns:
			Tupla di 3 tensori [B, hidden_dim, H/8, W/8], [B, hidden_dim, H/16, W/16], [B, hidden_dim, H/32, W/32]
		"""
		B, C, H, W = x.shape
		

		base_h = H // self.patch_size
		base_w = W // self.patch_size
		
		target_sizes = [
            (base_h * 4, base_w * 4),   # 1/8
            (base_h * 2, base_w * 2),   # 1/16
            (base_h, base_w),           # 1/32
        ]
		# Estrai feature intermedie dal ViT
		vit_feats = self._extract_intermediate_features(x)
		
		# Assicurati di avere almeno 3 livelli, ripeti se necessario (come DINOv3STAs)
		if len(vit_feats) == 1:
			vit_feats = [vit_feats[0], vit_feats[0], vit_feats[0]]
		elif len(vit_feats) == 2:
			vit_feats = [vit_feats[0], vit_feats[0], vit_feats[1]]
		
		# Prendi gli ultimi 3 livelli se ce ne sono di più
		vit_feats = vit_feats[-3:]
		
		# Resize alle dimensioni target (1/8, 1/16, 1/32)
		# Questo funziona correttamente sia per patch_size=14 che 16
		sem_feats = []
		for feat, target_size in zip(vit_feats, target_sizes):
			sem_feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
			sem_feats.append(sem_feat)
		
		# Fusione con feature spaziali dalla CNN (STA)
		fused_feats = []
		if self.use_sta and self.sta is not None:
			detail_feats = self.sta(x)  # Produce feature a scale 1/8, 1/16, 1/32
			for sem_feat, detail_feat in zip(sem_feats, detail_feats):
				# Allinea detail_feat a sem_feat se necessario
				if detail_feat.shape[-2:] != sem_feat.shape[-2:]:
					detail_feat = F.interpolate(detail_feat, size=sem_feat.shape[-2:], mode='bilinear', align_corners=False)
				fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
		else:
			fused_feats = sem_feats
		
		# Proiezione finale con conv 1x1 + norm
		c2 = self.norms[0](self.convs[0](fused_feats[0]))  # 1/8
		c3 = self.norms[1](self.convs[1](fused_feats[1]))  # 1/16
		c4 = self.norms[2](self.convs[2](fused_feats[2]))  # 1/32
		
		return c2, c3, c4


class RadioDinoSegModel(nn.Module):
	"""Modello completo per segmentazione con RadioDino/DINOv2 + STA + U-Net head.
	
	Architettura:
	1. RadioDinoSTAs: backbone ViT + CNN parallela per feature multi-scala
	2. UNetSegmentationHead: decoder U-Net-like per produrre le mask
	
	Questo design è ispirato a DEIMv2 e ottimizzato per segmentazione medica.
	"""

	def __init__(
		self,
		model_name: str,
		num_classes: int,
		image_size: Tuple[int, int],
		strategy: str = "native",  # Mantenuto per compatibilità, ora usa STA
		hidden_dim: Optional[int] = None,
		use_sta: bool = True,
		conv_inplane: int = 16,
		out_indices: Tuple[int, ...] = (2, 5, 8, 11),
		use_sync_bn: bool = False,
	) -> None:
		"""
		Args:
			model_name: Nome del modello timm (es. "hf_hub:Snarcy/RadioDino-s8")
			num_classes: Numero di classi per la segmentazione
			image_size: Dimensione input (H, W) per training/inference
			strategy: Deprecato, mantenuto per compatibilità
			hidden_dim: Dimensione hidden per encoder e decoder
			use_sta: Se True, usa SpatialPriorModule per dettagli locali
			conv_inplane: Canali base per SpatialPriorModule
			out_indices: Indici dei layer ViT da cui estrarre feature
			use_sync_bn: Se True, usa SyncBatchNorm (per multi-GPU)
		"""
		super().__init__()

		self.image_size = image_size
		
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
		
		# I canali di output sono tutti hidden_dim
		in_channels = self.backbone.out_channels
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
		# Estrai feature multi-scala con STA
		c2, c3, c4 = self.backbone(x)
		
		# Passa al decoder (ordine: alta risoluzione -> bassa risoluzione)
		feats = [c2, c3, c4]
		out = self.seg_head(feats, targets=None)
		return out
