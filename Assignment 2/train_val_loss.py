# just and extra script to compute the train and validation loss as i did not comput it after training

import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

class RobertaClassifier(nn.Module):
    def __init__(self, dropout: float, pooling: str):
        super().__init__()
        self.pooling = pooling
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / attention_mask.sum(1, keepdim=True)

        return self.classifier(self.dropout(pooled))

# to load best configuration:

with open("best_config.json") as f:
    best_config = json.load(f)

# to load data: 

def load_split(name):
    return (
        torch.load(f"tokenized_files/{name}_inputs.pt"),
        torch.load(f"tokenized_files/{name}_masks.pt"),
        torch.load(f"tokenized_files/{name}_labels.pt"),
    )

train_inputs, train_masks, train_labels = load_split("train")
val_inputs, val_masks, val_labels = load_split("val")

train_loader = DataLoader(
    TensorDataset(train_inputs, train_masks, train_labels),
    batch_size=best_config["batch_size"],
    shuffle=False,
)

val_loader = DataLoader(
    TensorDataset(val_inputs, val_masks, val_labels),
    batch_size=best_config["batch_size"],
    shuffle=False,
)

# to load best model:

model = RobertaClassifier(
    dropout=best_config["dropout"],
    pooling=best_config["pooling"]
).to(DEVICE)

model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()

# Loss computation:

loss_fn = nn.CrossEntropyLoss()

def compute_loss(loader):
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for ids, masks, labels in loader:
            ids, masks, labels = ids.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
            logits = model(ids, masks)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * ids.size(0)
            total_samples += ids.size(0)

    return total_loss / total_samples

train_loss = compute_loss(train_loader)
val_loss = compute_loss(val_loader)

print(f"Final Train Loss (Best Model): {train_loss:.4f}")
print(f"Final Val Loss   (Best Model): {val_loss:.4f}")

# save to csv:

loss_df = pd.DataFrame([{
    "model": "best_model",
    "train_loss": train_loss,
    "val_loss": val_loss
}])

loss_df.to_csv("train_val_results/best_model_loss_summary.csv", index=False)
print("Saved loss CSV to train_val_results/best_model_loss_summary.csv")


torch.save(
    {
        "train_loss": torch.tensor(train_loss),
        "val_loss": torch.tensor(val_loss)
    },
    "best_model_loss.pt"
)

# plot:

plt.figure()
plt.bar(["Train Loss", "Validation Loss"], [train_loss, val_loss])
plt.title("Loss Comparison of the Best model after training")
plt.ylabel("Cross-Entropy Loss")
plt.savefig("train_val_plots/best_model_loss_bar.png", bbox_inches="tight")
plt.show()
