import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .utils import get_activation
from .deim import DEIM


class DoubleConv(nn.Module):
	"""(Conv -> BN -> Act) * 2, stile U-Net."""

	def __init__(self, in_channels: int, out_channels: int, act: str = "relu") -> None:
		super().__init__()
		activation = get_activation(act)
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			activation,
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			activation,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
		return self.block(x)


class UpBlock(nn.Module):
	"""Blocco di upsampling U-Net: upsample + concat con skip + DoubleConv."""

	def __init__(
		self,
		in_channels: int,
		skip_channels: int,
		out_channels: int,
		act: str = "relu",
	) -> None:
		super().__init__()
		self.conv = DoubleConv(in_channels + skip_channels, out_channels, act=act)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
		# porta x alla stessa risoluzione spaziale dello skip
		x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
		x = torch.cat([x, skip], dim=1)
		return self.conv(x)


@register()
class UNetSegmentationHead(nn.Module):
	"""Decoder di segmentazione tipo U-Net.

	Assume una lista di feature multi-scala dall'encoder (es. `HybridEncoder`),
	ordinata da risoluzione più alta a più bassa, e risale concatenando skip
	connections fino alla scala più fine, producendo logits di segmentazione.
	
	Per segmentazione medica con input 512x512, se le feature partono da
	risoluzioni tipo 32x32, 16x16, 8x8 (tipico di ViT con patch_size=16),
	il decoder risale fino a 32x32 e poi fa upsample alla dimensione target.
	"""

	__share__ = ["num_classes", "eval_spatial_size"]

	def __init__(
		self,
		in_channels,
		num_classes: int = 21,
		hidden_dim: int | None = None,
		act: str = "relu",
		final_upsample: bool = False,
		eval_spatial_size=None,
		upsample_mode: str = "bilinear",
	) -> None:
		"""Args:
			in_channels: lista dei canali in ingresso per ogni scala
						(stesso ordine delle feature dall'encoder).
			num_classes: numero di classi per la segmentazione.
			hidden_dim: dimensione canali interna; se None usa in_channels[0].
			act: funzione di attivazione (stringa compatibile con get_activation).
			final_upsample: se True, fa un upsample finale a eval_spatial_size
							se disponibile; altrimenti lascia la risoluzione
							della scala più fine dell'encoder.
			eval_spatial_size: dimensione (H, W) opzionale per l'upsample finale.
			upsample_mode: modalità di interpolazione ("bilinear" o "nearest").
		"""

		super().__init__()

		if not isinstance(in_channels, (list, tuple)) or len(in_channels) < 1:
			raise ValueError("in_channels deve essere una lista/tupla non vuota")

		self.in_channels = list(in_channels)
		self.num_scales = len(self.in_channels)
		self.num_classes = num_classes
		self.hidden_dim = hidden_dim or self.in_channels[0]
		self.act = act
		self.final_upsample = final_upsample
		self.eval_spatial_size = eval_spatial_size
		self.upsample_mode = upsample_mode

		# Proiezione di tutti i livelli alla stessa dimensione di canali
		self.input_proj = nn.ModuleList(
			[
				nn.Conv2d(c_in, self.hidden_dim, kernel_size=1, bias=False)
				for c_in in self.in_channels
			]
		)

		# Blocchi di upsampling: da scala più bassa a più alta (es. 3->2->1)
		self.up_blocks = nn.ModuleList()
		for _ in range(self.num_scales - 1):
			self.up_blocks.append(
				UpBlock(
					in_channels=self.hidden_dim,
					skip_channels=self.hidden_dim,
					out_channels=self.hidden_dim,
					act=self.act,
				)
			)

		# Layer finale per produrre i logits di classe
		self.classifier = nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1)

	def forward(self, feats, targets=None):  # type: ignore[override]
		"""Args:
			feats: lista di feature maps [f0, f1, ..., fN],
				   da risoluzione più alta (f0) a più bassa (fN).
			targets: non usato qui, mantenuto per compatibilità.

		Returns:
			dict con chiave "pred_masks" contenente i logits
			di dimensione [B, num_classes, H, W].
		"""

		if not isinstance(feats, (list, tuple)):
			raise TypeError("feats deve essere una lista o tupla di tensori")
		if len(feats) != self.num_scales:
			raise ValueError(f"Attesi {self.num_scales} livelli di feature, ricevuti {len(feats)}")

		# Proietta tutte le feature alla stessa dimensione di canali
		proj_feats = [proj(f) for proj, f in zip(self.input_proj, feats)]

		# Partiamo dalla scala più bassa (ultima)
		x = proj_feats[-1]

		# Risaliamo concatenando gli skip delle scale più fini
		up_block_idx = 0
		for level in range(self.num_scales - 2, -1, -1):
			skip = proj_feats[level]
			x = self.up_blocks[up_block_idx](x, skip)
			up_block_idx += 1

		logits = self.classifier(x)

		# Upsample opzionale alla dimensione desiderata (se specificata)
		if self.final_upsample and self.eval_spatial_size is not None:
			h, w = self.eval_spatial_size
			logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

		return {"pred_masks": logits}


class SmallSegBackbone(nn.Module):
	"""Backbone molto piccolo che produce 3 livelli di feature.

	È pensato solo per testare rapidamente la testa di segmentazione
	e la pipeline DEIM end-to-end.
	"""

	def __init__(self, in_channels: int = 3, base_channels: int = 32, act: str = "relu") -> None:
		super().__init__()
		activation = get_activation(act)

		# livello 0: risoluzione più alta
		self.layer0 = nn.Sequential(
			nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(base_channels),
			activation,
		)

		# livello 1
		self.layer1 = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(base_channels * 2),
			activation,
		)

		# livello 2 (più grossolano)
		self.layer2 = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(base_channels * 4),
			activation,
		)

	def forward(self, x: torch.Tensor):  # type: ignore[override]
		f0 = self.layer0(x)
		f1 = self.layer1(f0)
		f2 = self.layer2(f1)
		# ordine: da risoluzione più alta a più bassa
		return [f0, f1, f2]


class IdentityEncoder(nn.Module):
	"""Encoder fittizio: passa attraverso le feature così come sono.

	Serve solo a rispettare l'interfaccia di `DEIM`.
	"""

	def __init__(self, in_channels=None, feat_strides=None, **kwargs) -> None:  # noqa: D401
		super().__init__()
		self.in_channels = in_channels or [32, 64, 128]
		self.feat_strides = feat_strides or [4, 8, 16]

	def forward(self, feats):  # type: ignore[override]
		return feats


def build_small_deim_segmentation(num_classes: int = 4, act: str = "relu") -> nn.Module:
	"""Costruisce un piccolo modello DEIM con testa di segmentazione.

	Usa `SmallSegBackbone` + `IdentityEncoder` + `UNetSegmentationHead`.
	"""

	base_channels = 32
	backbone = SmallSegBackbone(in_channels=3, base_channels=base_channels, act=act)

	# canali prodotti dal backbone ai tre livelli
	in_channels = [base_channels, base_channels * 2, base_channels * 4]
	encoder = IdentityEncoder(in_channels=in_channels, feat_strides=[4, 8, 16])

	head = UNetSegmentationHead(
		in_channels=in_channels,
		num_classes=num_classes,
		hidden_dim=base_channels * 4,
		act=act,
		final_upsample=False,
	)

	model = DEIM(backbone=backbone, encoder=encoder, decoder=head)
	return model


def test_small_deim_segmentation():
	"""Esegue un forward + calcolo loss per verificare che tutto funzioni.

	Usa input e target fittizi.
	"""

	num_classes = 4
	model = build_small_deim_segmentation(num_classes=num_classes)
	model.train()

	# input fittizio
	x = torch.randn(2, 3, 128, 128)
	# target di segmentazione [B, H, W] con label in [0, num_classes-1]
	target = torch.randint(0, num_classes, (2, 128, 128), dtype=torch.long)

	out = model(x)
	logits = out["pred_masks"]  # [B, C, H, W]

	# CrossEntropyLoss pixel-wise
	loss = F.cross_entropy(logits, target)
	print("logits shape:", logits.shape)
	print("loss:", loss.item())

	return loss


