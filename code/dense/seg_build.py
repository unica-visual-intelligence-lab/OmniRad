import argparse
from typing import List, Sequence

import torch

from engine.core import YAMLConfig
from engine.core.workspace import create
from engine.deim.deim import DEIM
from engine.deim.segmentation import UNetSegmentationHead


def build_deimv2_segmentation_model(
    cfg_path: str,
    num_classes: int,
    act: str = "relu",
    final_upsample: bool = True,
    eval_spatial_size=None,
) -> DEIM:
    """Costruisce un modello DEIMv2 con testa di segmentation U-Net-like.

    Questo funziona per *tutte* le taglie DEIMv2 (Atto, Femto, Pico, N, S, M, L, X)
    purché tu passi il relativo file di config `deimv2_*.yml`.
    Riusa backbone + HybridEncoder dal config e sostituisce il decoder
    di detection (`DEIMTransformer`) con `UNetSegmentationHead`.
    """

    cfg = YAMLConfig(cfg_path)
    # Override opzionale della eval_spatial_size (es. per segmentazione 512x512)
    if eval_spatial_size is not None:
        cfg.yaml_cfg["eval_spatial_size"] = list(eval_spatial_size)
    global_cfg = cfg.global_cfg

    # Config della sezione DEIM (definita in configs/base/deimv2.yml + override)
    if "DEIM" not in global_cfg:
        raise ValueError("Config YAML non contiene la sezione 'DEIM'.")
    deim_cfg = global_cfg["DEIM"]

    backbone_name = deim_cfg.get("backbone", "HGNetv2")
    encoder_name = deim_cfg.get("encoder", "HybridEncoder")

    # Istanzia backbone ed encoder come fa il sistema originale
    backbone = create(backbone_name, global_cfg)
    encoder = create(encoder_name, global_cfg)

    # Ricava i canali di uscita dall'encoder (generico per tutte le taglie)
    if hasattr(encoder, "out_channels") and isinstance(encoder.out_channels, Sequence):
        in_channels: List[int] = list(encoder.out_channels)
    else:
        # fallback: usa la configurazione di HybridEncoder dal global_cfg
        if "HybridEncoder" in global_cfg and "in_channels" in global_cfg["HybridEncoder"]:
            in_channels = list(global_cfg["HybridEncoder"]["in_channels"])
        else:
            raise ValueError(
                "Impossibile determinare in_channels per la segmentation head; "
                "assicurati che HybridEncoder abbia 'out_channels' o che il config "
                "contenga 'HybridEncoder.in_channels'."
            )

    eval_spatial_size = cfg.yaml_cfg.get("eval_spatial_size", None)

    seg_head = UNetSegmentationHead(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=in_channels[-1],  # usa l'ultimo livello come dimensione interna
        act=act,
        final_upsample=final_upsample,
        eval_spatial_size=eval_spatial_size,
    )

    model = DEIM(backbone=backbone, encoder=encoder, decoder=seg_head)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Costruisce un DEIMv2 con testa di segmentation.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Percorso al file di config deimv2_*.yml (qualsiasi taglia)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Numero di classi per la segmentazione.",
    )
    parser.add_argument(
        "--no-final-upsample",
        action="store_true",
        help="Disabilita l'upsample finale alla 'eval_spatial_size' del config.",
    )
    parser.add_argument(
        "--test-forward",
        action="store_true",
        help="Esegue un forward di test con input random per verificare le shape.",
    )
    args = parser.parse_args()

    model = build_deimv2_segmentation_model(
        cfg_path=args.config,
        num_classes=args.num_classes,
        final_upsample=not args.no_final_upsample,
    )

    if args.test_forward:
        model.eval()
        # Determina dimensione input: usa eval_spatial_size se definita, altrimenti 640x640
        from typing import Tuple

        spatial = args.config  # placeholder per type checker
        del spatial

        size = (640, 640)
        if hasattr(model, "encoder") and hasattr(model.encoder, "eval_spatial_size"):
            if model.encoder.eval_spatial_size is not None:
                size = tuple(model.encoder.eval_spatial_size)  # type: ignore[arg-type]
        elif "eval_spatial_size" in model.backbone.__dict__:
            # fallback molto conservativo
            size = tuple(model.backbone.eval_spatial_size)  # type: ignore[arg-type]

        h, w = size
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out = model(x)
        if "pred_masks" not in out:
            raise RuntimeError("Il modello non restituisce 'pred_masks' nel dict di output.")
        print("Input size:", x.shape)
        print("Pred masks size:", out["pred_masks"].shape)


if __name__ == "__main__":
    main()
