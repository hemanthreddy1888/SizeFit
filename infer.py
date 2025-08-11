import os
import torch
import numpy as np

from utils import load_config_from_json, to_var
from model import SFNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(saved_model_path, checkpoint):
    # Load model config
    model_config = load_config_from_json(
        os.path.join(saved_model_path, "config.jsonl")
    )

    # Init model
    model = SFNet(model_config["sfnet"])
    model = model.to(device)

    # Load weights
    checkpoint_path = os.path.join(saved_model_path, checkpoint)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, model_config

def prepare_single_input(user_id, cup_size, user_numeric, item_id, category, item_numeric):
    # Convert single datapoint into the same tensor format as training
    batch = {
        "user_id": torch.tensor([user_id], dtype=torch.int64),
        "cup_size": torch.tensor([cup_size], dtype=torch.int64),
        "user_numeric": torch.tensor([user_numeric], dtype=torch.float32),
        "item_id": torch.tensor([item_id], dtype=torch.int64),
        "category": torch.tensor([category], dtype=torch.int64),
        "item_numeric": torch.tensor([item_numeric], dtype=torch.float32),
    }
    # Move to GPU if available
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = to_var(v)
    return batch

if __name__ == "__main__":
    # Paths
    saved_model_path = "runs/trial_"  # your model path
    checkpoint = "model.pytorch"                          # change if needed

    # Load model
    model, model_config = load_model(saved_model_path, checkpoint)

    # Example new data (must match preprocessing: IDs & normalized numerics)
    new_data = [
        # (user_id, cup_size, [waist, hips, bra_size, height, shoe_size], item_id, category, [size, quality])
        (10, 2, [28, 36, 34, 65, 7], 501, 3, [4, 3]),
        (22, 1, [26, 34, 32, 62, 6], 640, 5, [2, 4]),
    ]

    for entry in new_data:
        batch = prepare_single_input(*entry)
        with torch.no_grad():
            logits, pred_probs = model(batch)
            pred_class = torch.argmax(pred_probs, dim=-1).item()
            print(f"Input: {entry}")
            print(f"Predicted class: {pred_class}  (0=small, 1=fit, 2=large)")
            print(f"Probabilities: {pred_probs.cpu().numpy()}")
            print("-" * 50)
