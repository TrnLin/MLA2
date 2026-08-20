# Raw data

Keep supplied source datasets here and never modify them.

Expected local layout:

```text
teacher/
  train/
    styles_train.csv
    images_train/
  test/
    styles_prediction.csv
    images_test/
```

The preparation code reads only these paths. Local symbolic links are allowed so
the large dataset does not need to be copied. Dataset files and links are ignored
by Git.
