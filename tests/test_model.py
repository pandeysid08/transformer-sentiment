import torch
from src.model import TransformerClassifier
from src.data import tokenize, build_vocab, encode


def test_forward_shape():
    m = TransformerClassifier(vocab_size=100, max_len=32)
    assert m(torch.randint(1, 100, (4, 32))).shape == (4, 2)


def test_padding_is_ignored():
    m = TransformerClassifier(vocab_size=100, max_len=16, dropout=0.0).eval()
    a = torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0]])
    b = torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert torch.allclose(m(a), m(b), atol=1e-5)


def test_tokenizer_and_vocab():
    v = build_vocab(["Great movie!", "great film", "bad movie"], min_freq=1)
    ids = encode("great unknownword", v, 5)
    assert len(ids) == 5 and ids[1] == 1 and ids[2:] == [0, 0, 0]
    assert tokenize("A<br />B") == ["a", "b"]


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    m = TransformerClassifier(vocab_size=50, d_model=32, n_layers=1, d_ff=64, max_len=8, dropout=0.0)
    x = torch.randint(1, 50, (8, 8)); y = torch.tensor([0, 1] * 4)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1
