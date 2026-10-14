import pytest

from favi_hub.evaluate import classification_metrics


def test_hand_calculated_metrics_and_fixed_label_order():
    result = classification_metrics([0, 1, 2], [0, 0, 2])
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(5 / 9)
    assert result["coverage_complete"]


def test_missing_class_is_reported():
    result = classification_metrics([0], [0])
    assert not result["coverage_complete"]
    assert result["macro_f1"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(("target", "prediction"), [([], []), ([0], []), ([3], [0]), ([True], [0])])
def test_invalid_labels_rejected(target, prediction):
    with pytest.raises(ValueError):
        classification_metrics(target, prediction)


def test_binary_nsfw_metrics_use_two_class_coverage():
    labels = {"Non-NSFW": 0, "NSFW": 1}
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], labels)
    assert result["labels"] == labels
    assert len(result["confusion_matrix"]) == 2
    assert result["coverage_complete"]
