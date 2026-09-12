import torch

from models import DecoderLSTM, EncoderLSTM, Seq2SeqLSTM, SequencePredictor, VisualAutoencoder


def test_visual_autoencoder_shapes():
    content, context = VisualAutoencoder(latent_dim=8)(torch.rand(2, 3, 60, 125))
    assert content.shape == (2, 3, 60, 125)
    assert context.shape == content.shape


def test_sequence_predictor_shapes():
    vocab_size = 31
    text = Seq2SeqLSTM(EncoderLSTM(vocab_size, 8, 8), DecoderLSTM(vocab_size, 8, 8))
    model = SequencePredictor(VisualAutoencoder(latent_dim=8), text, latent_dim=8, gru_hidden_dim=8)
    frames = torch.rand(2, 4, 3, 60, 125)
    descriptions = torch.randint(vocab_size, (2, 4, 12))
    target = torch.randint(vocab_size, (2, 1, 12))
    image, context, logits, *_ = model(frames, descriptions, target)
    assert image.shape == (2, 3, 60, 125)
    assert context.shape == image.shape
    assert logits.shape == (2, 11, vocab_size)