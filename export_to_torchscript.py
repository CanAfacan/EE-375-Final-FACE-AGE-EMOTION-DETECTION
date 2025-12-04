from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from train_fer2013 import Config, build_model, EMOTIONS  
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/best_model.pth")
    ap.add_argument("--out",  default="model.torchscript.pt")
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    cfg_d = ckpt.get("config", {}) or {}
    label_map = ckpt.get("label_map", {i: n for i, n in enumerate(EMOTIONS)})
    cfg = Config(model=cfg_d.get("model", "simple_cnn"), img_size=int(cfg_d.get("img_size", 48) or 48))
    model = build_model(cfg, num_classes=len(label_map))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    in_ch = 3 if cfg.model == "resnet18" else 1
    example = torch.randn(1, in_ch, cfg.img_size, cfg.img_size)
    ts = torch.jit.trace(model, example)
    ts.save(args.out)
    with open(Path(args.out).with_suffix(".labels.json"), "w") as f:
        json.dump({int(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"[Saved] {args.out}")
    print("Labels:", [label_map[i] for i in range(len(label_map))])
if __name__ == "__main__":
    main()
