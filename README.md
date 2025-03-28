# MDA_for_Face_Recognition

Duplicate paper: Multilinear Discriminant Analysis for Face Recognition
<p align="center">
  <img src="Image/Paper_title.png" width="600">
</p>

### Dataset

**ORL:**

The [ORL database](https://drive.google.com/file/d/1UKyCXY6UJxIK4aMlPbBX_dVtgxnqEg1K/view?usp=drive_link)
contains 400 images of 40 individuals. These images were captured at different times and have different variations
including expression
(open or closed eyes, smiling or nonsmiling) and facial details (glasses or no glasses). The images were taken with a
tolerance for some tilting and rotation of the face up to 20.
All images were in grayscale and normalized to the resolution of pixels and histogram equilibrium was applied in the
preprocessing step.

**FERET**

Two types of experiments were conducted on
the [FERET database](https://U9010953:Xf8Gr5Nk93@nigos.nist.gov/colorferet/colorferet.tar).   
One is conducted on 70 people of the FERET database with six different images for each person;   
two of them were applied as gallery set and the other four for probe set.   
We extracted 40 Gabor features with five different scales and eight different directions in the down-sampled positions
and each image was encoded as a third-order tensor of size for MDA/3-3.

### MDA process

<p align="center">
  <img src="Image/MDA_process.png" width="400">
</p>

### Experience Result

#### For ORL:

**Eigenface**

|                        | G5/P5 | G4/P6 | G3/P7 | G2/P8 |
|:----------------------:|:-----:|:-----:|:-----:|:-----:|
|        Accuracy        | 0.82  | 0.71  | 0.69  | 0.60  |
| Time consume <br/>(ms) | 0.05  | 0.04  | 0.03  | 0.04  |

**Fisherface**

|                        | G5/P5 | G4/P6 | G3/P7 | G2/P8 |
|:----------------------:|:-----:|:-----:|:-----:|:-----:|
|        Accuracy        | 0.90  | 0.85  | 0.80  | 0.68  |
| Time consume <br/>(ms) |  129  |  119  |  117  |  121  |

**LDA**

|                        | G5/P5 | G4/P6 | G3/P7 | G2/P8 |
|:----------------------:|:-----:|:-----:|:-----:|:-----:|
|        Accuracy        | 0.88  | 0.80  | 0.76  | 0.55  |
| Time consume <br/>(ms) | 0.25  | 0.18  | 0.14  | 0.08  |

**MDA**

|                        | G5/P5 | G4/P6 | G3/P7 | G2/P8 |
|:----------------------:|:-----:|:-----:|:-----:|:-----:|
|        Accuracy        | 0.92  | 0.90  | 0.79  | 0.69  |
| Time consume <br/>(ms) |  28   |  24   |  23   |  19   |
