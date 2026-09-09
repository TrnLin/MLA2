from fashion.task1.classical import TASK1_HOG_COARSE, TASK1_HOG_FINE
from fashion.task1.image_contract import (
    TASK1_IMAGE_SIZE,
    TASK1_PAD_COLOR,
    TASK1_TENSOR_SHAPE,
)
from fashion.task1.preprocessing import (
    DEFAULT_TASK1_PREPROCESSING,
    TASK1_CONTROL_PREPROCESSING,
)


def test_task1_models_share_one_image_geometry() -> None:
    assert TASK1_IMAGE_SIZE == (80, 60)
    assert TASK1_PAD_COLOR == (255, 255, 255)
    assert TASK1_TENSOR_SHAPE == (3, 80, 60)
    assert DEFAULT_TASK1_PREPROCESSING.image_size == TASK1_IMAGE_SIZE
    assert TASK1_CONTROL_PREPROCESSING.image_size == TASK1_IMAGE_SIZE
    assert DEFAULT_TASK1_PREPROCESSING.pad_color == TASK1_PAD_COLOR
    assert TASK1_CONTROL_PREPROCESSING.pad_color == TASK1_PAD_COLOR
    assert TASK1_HOG_COARSE.image_size == TASK1_IMAGE_SIZE
    assert TASK1_HOG_FINE.image_size == TASK1_IMAGE_SIZE
    assert TASK1_HOG_COARSE.pad_color == TASK1_PAD_COLOR
    assert TASK1_HOG_FINE.pad_color == TASK1_PAD_COLOR
