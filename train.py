"""Train the sequence predictor on StoryReasoning."""

import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from data import SequencePredictionDataset
from models import DecoderLSTM, EncoderLSTM, Seq2SeqLSTM, SequencePredictor, VisualAutoencoder
from training import init_weights, save_checkpoint, sequence_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", default="checkpoints/sequence_predictor.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    dataset = load_dataset("daniel3303/StoryReasoning", split="train")
    sequence_dataset = SequencePredictionDataset(dataset, tokenizer)
    train_size = int(0.8 * len(sequence_dataset))
    train_dataset, _ = random_split(sequence_dataset, [train_size, len(sequence_dataset) - train_size])
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    text = Seq2SeqLSTM(EncoderLSTM(tokenizer.vocab_size, 16, 16), DecoderLSTM(tokenizer.vocab_size, 16, 16))
    visual = VisualAutoencoder()
    visual.apply(init_weights)
    model = SequencePredictor(visual, text).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = tuple(value.to(device) if torch.is_tensor(value) else value for value in batch)
            optimizer.zero_grad()
            losses = sequence_loss(model(batch[0], batch[1], batch[3]), batch,
                                   tokenizer.pad_token_id, model.image_encoder)
            losses["total"].backward()
            optimizer.step()
            epoch_loss += losses["total"].item()
        epoch_loss /= max(1, len(loader))
        print(f"epoch={epoch + 1}/{args.epochs} loss={epoch_loss:.4f}")
        save_checkpoint(args.checkpoint, model, optimizer, epoch + 1, epoch_loss)


if __name__ == "__main__":
    main()