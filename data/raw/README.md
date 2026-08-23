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

Optional external data is kept separate:

```text
external/
  fashion_product_images_v1/
    images/
    images.csv
```

Shared preparation never scans or parses this external collection. It is not
uniformly 4K and it contains both teacher train and official prediction IDs, so it
must not be named or treated as a second teacher training set. A task owner may use
it only through a later, explicit, documented experiment.
