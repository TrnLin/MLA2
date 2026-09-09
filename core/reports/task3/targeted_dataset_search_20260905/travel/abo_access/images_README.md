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

The `images/` directory, `abo-images-original.tar` and `abo-images-small.tar`
archives are made of the following files:

  * `LICENSE-CC-BY-4.0.txt` - The License file. You must read, agree and
    comply to the License before using the Amazon Berkeley Objects data.

  * `images/metadata/images.csv.gz` - Image metadata. This file is a
    gzip-compressed comma-separated value (CSV) file with the following
    columns: `image_id`, `height`, `width`, and `path`.
    - `image_id` (string): this id uniquely refers to a product image. This id
      can be used to retrieve the image data from Amazon's Content Delivery
      Network (CDN) using the template:
      `https://m.media-amazon.com/image/I/<image_id>.<extension>` [^1],
      where `<extension>` is composed of the characters following the dot in the
      `path` field. Any value occurring in the `main_image` and `other_images`
      attributes of product metadata is an `image_id` present in this file.
    - `height` (int) and `width` (int): respectively, the height and width of
      the original image.
    - `path`: the location of the image file relative to the `images/original/`
      or `images/small/` directories. A path is composed of lowercase hex
      characters (`0-9a-f`) that also uniquely identifies images. The first two
      characters are used to build a file hierarchy and reduce the number of
      images in a single directory. The extension is `jpg` except for few `png`
      files.
    
    Below are are first 10 lines of `images/metadata/images.csv.gz`:
```
image_id,height,width,path
010-mllS7JL,106,106,14/14fe8812.jpg
01dkn0Gyx0L,122,122,da/daab0cad.jpg
01sUPg0387L,111,111,d2/d2daaae9.jpg
1168jc-5r1L,186,186,3a/3a4e88e6.jpg
11RUV5Fs65L,30,500,d9/d91ab9cf.jpg
11X4pFHqYOL,35,500,20/20098c4d.jpg
11Y+Xpt1lfL,103,196,99/9987a1c8.jpg
11rL64ZLPYL,64,500,89/89a2ff4d.jpg
11xjmNF5TAL,117,88,ee/ee239f0f.jpg
```
      
  * `images/original/<path>` - Original image data. This directory contains the
     original high-resolution version of the images. See
     `images/metadata/images.csv.gz` for details of image naming.

  * `images/small/<path>` - Downscaled image data. This directory contains the
     version of the images, where they have been downscaled such that their
     largest axis (height or width) is a maximum of 256 pixels. See
     `images/metadata/images.csv.gz` for details of image naming.
  
## Footnotes

[^1]: Importantly, there is no guarantee that these URLs will remain unchanged
and available on the long term, we thus recommend using the images provided in
the archives instead.
