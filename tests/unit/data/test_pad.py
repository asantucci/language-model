import torch

from data.data_collator import pad  # adjust path as needed

def test_pad_basic():
    examples = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6]),
    ]
    padded = pad(examples, pad_value=0)

    assert all(len(tensor) == 3 for tensor in padded), "All sequences must be padded to length 3."
    assert torch.equal(padded[0], torch.tensor([1, 2, 3]))
    assert torch.equal(padded[1], torch.tensor([4, 5, 0]))
    assert torch.equal(padded[2], torch.tensor([6, 0, 0]))

def test_pad_different_pad_value():
    examples = [
        torch.tensor([1, 2]),
        torch.tensor([3]),
    ]
    padded = pad(examples, pad_value=-1)

    assert torch.equal(padded[0], torch.tensor([1, 2]))
    assert torch.equal(padded[1], torch.tensor([3, -1]))
