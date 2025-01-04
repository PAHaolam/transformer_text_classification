import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import torch
from shutil import copyfile
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import wandb
import pickle

from vocab import Vocab
from dataset import ViOCD_Dataset, collate_fn
from transformer_encoder_model import TransformerEncoderModel
from Pytorch_transformer_model import PyTorchTransformerEncoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"
scorers = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score
}

def train(epoch: int, model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer):
    model.train()

    running_loss = .0
    with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, items in enumerate(dataloader):
            # forward pass
            input_ids = items["input_ids"].to(device)
            labels = items["labels"].to(device)
            
            _, loss = model(input_ids, labels)
            
            # backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()

            # update the training status
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

def compute_scores(predictions: list, labels: list) -> dict:
    scores = {}
    for scorer_name in scorers:
        scorer = scorers[scorer_name]
        scores[scorer_name] = scorer(labels, predictions, average="macro")

    return scores

def evaluate_metrics(epoch: int, model: nn.Module, dataloader: DataLoader) -> dict:
    model.eval()
    all_labels = []
    all_predictions = []
    scores = {}
    with tqdm(desc='Epoch %d - Evaluating' % epoch, unit='it', total=len(dataloader)) as pbar:
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
        # Calculate metrics
    scores = compute_scores(all_predictions, all_labels)

    return scores

def save_checkpoint(dict_to_save: dict, checkpoint_path: str):
    torch.save(dict_to_save, os.path.join(f"{checkpoint_path}", "last_model.pth"))

def main(
        d_model: int = 512,
        layer_dim: int = 3,
        head: int = 8,
        d_ff: int = 4096,
        dropout: float = 0.1,
        train_path: str = "UIT-ViOCD/train.json", 
        dev_path: str = "UIT-ViOCD/dev.json", 
        test_path: str = "UIT-ViOCD/test.json",
        learning_rate: float = 0.001,
        checkpoint_path: str = "checkpoints",
        positional_encoding: str = "sinusoidal",
        model_type: str = "pytorch",
        seed: int = 42
    ):
    vocab = Vocab(
        train_path, dev_path, test_path
    )
    # Lưu vocab
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    vocab_path = os.path.join(checkpoint_path, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    train_dataset = ViOCD_Dataset(train_path, vocab)
    dev_dataset = ViOCD_Dataset(dev_path, vocab)
    test_dataset = ViOCD_Dataset(test_path, vocab)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    epoch = 0
    score_name = "f1"
    allowed_patience = 5
    best_score = 0

    # Tạo mô hình dựa trên `model_type`
    if model_type == "custom":
        use_learned_positional_encoding = (positional_encoding.lower() == "learned")
        model = TransformerEncoderModel(
            d_model, head, layer_dim, d_ff, dropout, vocab, use_learned_positional_encoding=use_learned_positional_encoding
        ).to(device)
    elif model_type == "pytorch":
        model = PyTorchTransformerEncoderModel(
            d_model, head, layer_dim, d_ff, dropout, vocab, seed
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))

    while True:
        train(epoch, model, train_dataloader, optim)
        # val scores
        train_scores = evaluate_metrics(epoch, model, train_dataloader)
        print(f"Train scores: {train_scores}")
        dev_scores = evaluate_metrics(epoch, model, dev_dataloader)
        print(f"Dev scores: {dev_scores}")
        test_scores = evaluate_metrics(epoch, model, test_dataloader)
        print(f"Test scores: {test_scores}")
        dev_score = dev_scores[score_name]

        wandb.log({
            "train/f1":train_scores["f1"],
            "train/precision":train_scores["precision"],
            "train/recall":train_scores["recall"],
            "dev/f1":dev_scores["f1"],
            "dev/precision":dev_scores["precision"],
            "dev/recall":dev_scores["recall"],
            "test/f1":test_scores["f1"],
            "test/precision":test_scores["precision"],
            "test/recall":test_scores["recall"],
        })

        # Prepare for next epoch
        is_the_best_model = False
        if dev_score > best_score:
            best_score = dev_score
            patience = 0
            is_the_best_model = True
        else:
            patience += 1

        exit_train = False

        if patience == allowed_patience:
            exit_train = True

        save_checkpoint({
            "epoch": epoch,
            "best_score": best_score,
            "patience": patience,
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict()
        }, checkpoint_path)

        if is_the_best_model:
            copyfile(
                os.path.join(checkpoint_path, "last_model.pth"), 
                os.path.join(checkpoint_path, "best_model.pth")
            )

        if exit_train:
            break

        epoch += 1

    scores = evaluate_metrics(epoch, model, test_dataloader)
    print(f"Test scores: {scores}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Encoder Training Script")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--layer_dim", type=int, default=3, help="Number of layers")
    parser.add_argument("--head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=4096, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train_path", type=str, default="UIT-ViOCD/train.json", help="Path to the training dataset")
    parser.add_argument("--dev_path", type=str, default="UIT-ViOCD/dev.json", help="Path to the dev dataset")
    parser.add_argument("--test_path", type=str, default="UIT-ViOCD/test.json", help="Path to the test dataset")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--positional_encoding", type=str, choices=["sinusoidal", "learned"], default="sinusoidal", help="Type of positional encoding")
    parser.add_argument("--model_type", type=str, choices=["custom", "pytorch"], default="pytorch", help="Model type to use")
    parser.add_argument('--wandb', default='disabled', type=str)
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    
    args = parser.parse_args()

    wandb.init(project='transformer_text_classification',
               config=args,
               mode=args.wandb)
    
    main(
        d_model=args.d_model,
        layer_dim=args.layer_dim,
        head=args.head,
        d_ff=args.d_ff,
        dropout=args.dropout,
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        learning_rate=args.learning_rate,
        checkpoint_path=f"{args.checkpoint_path}_seed_{args.seed}",
        positional_encoding=args.positional_encoding,
        model_type=args.model_type,  # Truyền model_type
        seed=args.seed
    )