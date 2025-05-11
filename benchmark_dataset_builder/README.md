# Benchmark Dataset Builder

> Flatten and annotate raw datasets like KITTI, Málaga, and Cityscapes into benchmark-ready formats with just one command.

---

## What is this?

**Benchmark Dataset Builder** is a collection of CLI scripts that simplify the process of:

- Flattening image folders from complex dataset structures  
- Sequentially renaming images (e.g., `Kitti1.png`, `Kitti2.png`, ...)  
- Attaching relevant camera metadata to each image in `.json` format  

It supports:

- **KITTI Raw Dataset** (urban, forest, and residential driving)  
- **Málaga Urban Dataset**  
- **Cityscapes Dataset**  

The output is standardized and easy to plug into training pipelines.

---

## Folder Structure (Output Example)

```text
Kitti-test-benchmark/
├── Kitti1.png
├── Kitti1.json
├── Kitti2.png
├── Kitti2.json
...
```

Each image is renamed sequentially and paired with a `.json` file containing metadata such as `fx`, `fy`, `cx`, `cy`, `k1`, `k2`, `p1`, `p2`, `k3` (if available).

---

## Quick Start

### KITTI Example

```bash
python kitti_flatten_with_metadata.py \
  /path/to/kitti_raw/ \
  /path/to/Kitti-test-benchmark \
  --prefix Kitti
```

### Málaga Example

```bash
python malaga_flatten_with_metadata.py \
  /path/to/malaga_dataset/ \
  /path/to/Malaga-test-benchmark \
  --prefix Malaga
```

### Cityscapes Example

```bash
python cityscapes_flatten_with_metadata.py \
  /path/to/cityscapes/ \
  /path/to/Cityscapes-test-benchmark \
  --prefix City
```

---

## Script Overview

| Script                              | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `kitti_flatten_with_metadata.py`     | Flattens KITTI Raw and writes cam_0 metadata (fx, fy, cx, cy, k1, k2, p1, p2, k3) |
| `malaga_flatten_with_metadata.py`    | Flattens Málaga sequences and includes available camera intrinsics         |
| `cityscapes_flatten_with_metadata.py`| Converts Cityscapes images into a flat format and attaches label info if needed |

---

## Dataset Citations

**KITTI Raw Dataset**  
Geiger, A., Lenz, P., & Urtasun, R. (2012). *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.*  
In Proceedings of CVPR.  
http://www.cvlibs.net/datasets/kitti/

**Málaga Urban Dataset**  
Blanco-Claraco, J. L., Moreno-Dueñas, F. A., & González-Jiménez, J. (2014). *The Málaga Urban Dataset: High-rate Stereo and Lidars in a realistic urban scenario.*  
The International Journal of Robotics Research, 33(2), 207–214.  
https://www.mrpt.org/MalagaDataset

**Cityscapes Dataset**  
Cordts, M., Omran, M., Ramos, S., et al. (2016). *The Cityscapes Dataset for Semantic Urban Scene Understanding.*  
In Proceedings of CVPR.  
https://www.cityscapes-dataset.com

---

## Contribute

Feel free to open issues or pull requests to support new datasets or improvements.

---

## Contact

Maintainer: [faiz@ailivesim.com](mailto:faiz@ailivesim.com)
