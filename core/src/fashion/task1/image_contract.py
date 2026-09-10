"""Shared deterministic image geometry for every Task 1 model family."""

TASK1_IMAGE_SIZE: tuple[int, int] = (80, 60)
TASK1_PAD_COLOR: tuple[int, int, int] = (255, 255, 255)
TASK1_TENSOR_SHAPE: tuple[int, int, int] = (3, *TASK1_IMAGE_SIZE)
