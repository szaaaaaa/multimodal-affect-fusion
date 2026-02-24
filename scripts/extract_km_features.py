import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch

# Ensure project root is on sys.path for local imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encoder.km.km_encoder_stat import KMStatEncoder
from encoder.km.km_encoder_1dCNN import KM1DCNNEncoder



def load_session_data(session_path):
    """从session文件夹读取三类CSV数据"""
    kb_file = session_path / "keyboard.csv"
    mb_file = session_path / "mousebuttons.csv"
    mp_file = session_path / "mouseposition.csv"

    data = {}
    if kb_file.exists():
        df = pd.read_csv(kb_file)
        data["keyboard"] = df.to_dict("records")
    if mb_file.exists():
        df = pd.read_csv(mb_file)
        data["mousebuttons"] = df.to_dict("records")
    if mp_file.exists():
        df = pd.read_csv(mp_file)
        data["mouseposition"] = df.to_dict("records")
    
    return data


def encode_all_sessions(
    root_dir,
    output_dir,
    dt=0.2,
    encoder_type: str = "stat",
    cnn_dim: int = 64,
    skip_existing: bool = True,
):
    """遍历所有被试和session，提取特征并保存"""
    root = Path(root_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    encoder = KMStatEncoder(dt=dt, device="cpu")
    cnn_encoder = None
    
    for subject_dir in sorted(root.glob("S*")):
        subject_id = subject_dir.name
        
        for session_dir in sorted(subject_dir.glob("P*")):
            session_id = session_dir.name
            
            print(f"Processing {subject_id}/{session_id}...")

            save_path = out / f"{subject_id}_{session_id}.pt"
            if skip_existing and save_path.exists():
                print(f"  Skipped (already exists): {save_path.name}")
                continue
            
            try:
                raw_data = load_session_data(session_dir)
                if not raw_data:
                    print(f"  Skipped (no data files)")
                    continue
                
                result = encoder.encode(raw_data)

                if encoder_type == "cnn":
                    feats = result["features"]
                    if cnn_encoder is None or cnn_encoder.conv1.in_channels != feats.shape[1]:
                        cnn_encoder = KM1DCNNEncoder(d_in=int(feats.shape[1]), d_model=cnn_dim)
                    with torch.no_grad():
                        feats_cnn = cnn_encoder(feats.unsqueeze(0)).squeeze(0)
                    result["features"] = feats_cnn
                    result["meta"]["feature_dim"] = int(feats_cnn.shape[1])
                    result["meta"]["feature_names"] = [f"cnn_{i}" for i in range(int(feats_cnn.shape[1]))]
                
                tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                torch.save(result, tmp_path)
                tmp_path.replace(save_path)
                print(f"  Saved to {save_path}")
                
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "data" / "features" / "amucs" / "km"),
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--encoder", type=str, default="stat", choices=["stat", "cnn"])
    parser.add_argument("--cnn_dim", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")
    args = parser.parse_args()

    encode_all_sessions(
        args.root_dir,
        args.output_dir,
        dt=args.dt,
        encoder_type=args.encoder,
        cnn_dim=args.cnn_dim,
        skip_existing=not args.overwrite,
    )
