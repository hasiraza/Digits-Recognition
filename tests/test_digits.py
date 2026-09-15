from sklearn.datasets import load_digits


def test_digits_dataset():
    digits = load_digits()

    assert digits.data.shape[1] == 64
    assert len(digits.target_names) == 10