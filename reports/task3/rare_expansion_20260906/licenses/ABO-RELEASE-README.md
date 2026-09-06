# Amazon Berkeley Objects (c) by Amazon.com

[Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)
is a collection of product listings with multilingual metadata, catalog
imagery, high-quality 3d models with materials and parts, and benchmarks derived
from that data.

## License

This work is licensed under the Creative Commons Attribution 4.0 International
Public License. To obtain a copy of the full license, see LICENSE-CC-BY-4.0.txt,
visit [CreativeCommons.org](https://creativecommons.org/licenses/by/4.0/)
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Under the following terms:

  * Attribution — You must give appropriate credit, provide a link to the
    license, and indicate if changes were made. You may do so in any reasonable
    manner, but not in any way that suggests the licensor endorses you or your
    use.

  * No additional restrictions — You may not apply legal terms or technological
    measures that legally restrict others from doing anything the license
    permits.
    
## Attribution

Credit for the data, including all images and 3d models, must be given to:

> Amazon.com

Credit for building the dataset, archives and benchmark sets must be given to:

> Matthieu Guillaumin (Amazon.com), Thomas Dideriksen (Amazon.com),
> Kenan Deng (Amazon.com), Himanshu Arora (Amazon.com),
> Jasmine Collins (UC Berkeley) and Jitendra Malik (UC Berkeley)

## Description

Amazon Berkeley Objects is a collection of 147,702 product listings with
multilingual metadata and 398,212 unique catalog images. 8,222 listings come
with turntable photography (also referred as *spin* or *360º-View* images), as
sequences of 24 or 72 images, for a total of 586,584 images in 8,209 unique
sequences. For 7,953 products, the collection also provides high-quality 3d
models, as glTF 2.0 files.

The collection is made of the following directories and files:

  * `README.md` - The present file.

  * `LICENSE-CC-BY-4.0.txt` - The License file. You must read, agree and comply
    to the License before using the Amazon Berkeley Objects data.

  * `listings/` - Product description and metadata. Check `listings/README.md`
    for details. `archives/abo-listings.tar` contains all the files in
    `listings/` as a tar archive.
  
  * `images/` - Catalog imagery, in original and smaller (256px) resolution.
    Check `images/README.md` for details. `archives/abo-images-original.tar`
    contains the metadata and original images from `images/original/` as a tar
    archive and `archives/abo-images-small.tar` contains the metadata and
    downscaled images from `images/small/` as a tar archive.
  
  * `spins/` - Spin / 360º-View images and metadata. Check `spins/README.md` for
    details. `archives/abo-spins.tar` contains the metadata and images from
    `spins/` as a tar archive.

  * `3dmodels/` - 3D models and metadata. Check `3dmodels/README.md` for details.
    `archives/abo-3dmodels.tar` contains the metadata and 3d models from 
    `3dmodels/` as a tar archive.

  * `benchmarks/abo-mvr.csv.xz` - Train/val/test dataset splits for the Multi-
    View Retrieval experiments of the [CVPR 2022 ABO paper](
    https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/static_html/ABO_CVPR2022.pdf)

  * `archives/abo-benchmark-material.tar` - Train/test dataset for the Material
    Prediction experiments of the [CVPR 2022 ABO paper](
    https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/static_html/ABO_CVPR2022.pdf).
    See the `README.md` file in the archive for more details.

  * `archives/abo-part-labels.tar` - Dataset for the [2023 ABO Fine-grained
    Semantic Segmentation Competition](
    https://eval.ai/web/challenges/challenge-page/2027/overview)
    organized for the [3D Vision and Modeling Challenges in eCommerce Workshop](
    https://3dv-in-ecommerce.github.io) in conjunction with ICCV 2023.

## Footnotes

[^1]: Importantly, there is no guarantee that these URLs will remain unchanged
and available on the long term, we thus recommend using the images provided in
the archives instead.
