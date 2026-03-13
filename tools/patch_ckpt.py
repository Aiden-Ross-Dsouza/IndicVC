import torch

for fname in ["checkpoints/speaker/last.pt",
              "checkpoints/speaker/best.pt",
              "checkpoints/speaker/best_eer.pt"]:
    ckpt = torch.load(fname, map_location="cpu", weights_only=False)
    ckpt["n_speakers"]   = 1938
    ckpt["wandb_run_id"] = "ihh8eodr"
    torch.save(ckpt, fname)
    print(f"Patched {fname}: epoch={ckpt['epoch']}, n_speakers={ckpt['n_speakers']}, wandb={ckpt['wandb_run_id']}")