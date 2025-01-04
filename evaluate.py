import os
import torch
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from vocab import Vocab
from dataset import ViOCD_Dataset, collate_fn
from Pytorch_transformer_model import PyTorchTransformerEncoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"
scorers = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score
}

def compute_scores(predictions: list, labels: list) -> dict:
    scores = {}
    for scorer_name in scorers:
        scorer = scorers[scorer_name]
        scores[scorer_name] = scorer(labels, predictions, average="macro")

    return scores

def evaluate_metrics(model: torch.nn.Module, dataloader: DataLoader) -> dict:
    model.eval()
    all_labels = []
    all_predictions = []
    with tqdm(desc='Evaluating', unit='it', total=len(dataloader)) as pbar:
        for items in dataloader:
            input_ids = items["input_ids"].to(device)
            labels = items["labels"].to(device)
            with torch.no_grad():
                logits, _ = model(input_ids, labels)

            predictions = logits.argmax(dim=-1).long()

            labels = labels.view(-1).cpu().numpy()
            predictions = predictions.view(-1).cpu().numpy()

            all_labels.extend(labels)
            all_predictions.extend(predictions)

            pbar.update()

    return compute_scores(all_predictions, all_labels)

def main(checkpoint_path: str, train_path: str, dev_path: str, test_path: str):
    # Load vocab
    vocab_path = os.path.join(checkpoint_path, "vocab.pkl")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}")

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Create datasets and dataloaders
    train_dataset = ViOCD_Dataset(train_path, vocab)
    dev_dataset = ViOCD_Dataset(dev_path, vocab)
    test_dataset = ViOCD_Dataset(test_path, vocab)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    best_model_path = os.path.join(checkpoint_path, "best_model.pth")
    if os.path.exists(best_model_path):
        print("Loading best model...")
        best_model = torch.load(best_model_path).to(device)
        print("Evaluating best model:")
        print("Train scores:", evaluate_metrics(best_model, train_dataloader))
        print("Dev scores:", evaluate_metrics(best_model, dev_dataloader))
        print("Test scores:", evaluate_metrics(best_model, test_dataloader))
    else:
        print("Best model not found.")

    last_model_path = os.path.join(checkpoint_path, "last_model.pth")
    if os.path.exists(last_model_path):
        print("Loading last model...")
        last_model = torch.load(last_model_path).to(device)
        print("Evaluating last model:")
        print("Train scores:", evaluate_metrics(last_model, train_dataloader))
        print("Dev scores:", evaluate_metrics(last_model, dev_dataloader))
        print("Test scores:", evaluate_metrics(last_model, test_dataloader))
    else:
        print("Last model not found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--train_path", type=str, default="UIT-ViOCD/train.json", help="Path to the training dataset")
    parser.add_argument("--dev_path", type=str, default="UIT-ViOCD/dev.json", help="Path to the dev dataset")
    parser.add_argument("--test_path", type=str, default="UIT-ViOCD/test.json", help="Path to the test dataset")

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint_path,
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path
    )
