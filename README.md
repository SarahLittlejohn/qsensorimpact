# qsensorimpact

`qsensorimpact` is a Python framework for **simulating, detecting, and evaluating rare-event impact signatures** in qubit-array data.  

The framework provides tools to:
- Generate synthetic impact signatures with configurable spatial and temporal characteristics  
- Detect candidate impact regions using **YOLOv5**  
- Analyse results with clustering and statistical methods  
- Evaluate accuracy and performance with standardised metrics  

Developed as part of MSc Quantum Technologies research at UCL / NPL.

---

## ⚙️ Setup

Clone the repository and install the dependencies into a virtual environment:

```bash
git clone https://github.com/SarahLittlejohn/qsensorimpact.git
cd qsensorimpact
pip install -r requirements.txt
```
To see examples on how to run the code, run specific regions in example_use_qsensorimpact.py.

## 📂 Generation

The `generation/` directory contains functions for **creating synthetic impact signatures** that mimic rare error events in qubit arrays.  

### Available models
- **Gaussian impacts** — smooth, spatially distributed signatures  
- **Delta impacts** — sharp, localised signatures  
- **Gamma-like (QP burst) impacts** — realistic quasiparticle-driven events  

### Configurable parameters
Each generator allows you to control:
- **Grid size** — number of qubits along each dimension  
- **Baseline switching rate** — steady-state value before impact  
- **Noise level** — background fluctuations  
- **Spatial spread (σ)** — how wide the impact is across the grid  
- **Temporal profile and duration** — shape and length of the impact in time  

### Outputs
- Generates **3D time-dependent tensors** representing qubit-array activity  
- These tensors serve as inputs to the analysis pipeline  

## 📂 Analysis

The `analysis/` directory implements the **analysis pipeline**, combining deep-learning detection and clustering.  

### Components
- **Preprocessing** — converts parity time series into switching-rate frames using HMMs  
- **YOLOv5 object detection** — identifies candidate impact regions within frame sequences  
- **Spatio-temporal clustering (DBSCAN)** — groups detections that are close in both space and time to reconstruct complete events  
- **Quartile analysis** — visualises and summarises which regions of the qubit array are most affected  

### Outputs
The analysis stage produces the **predicted spatial-temporal components** of each impact:
- **x** - horizontal location on the qubit grid  
- **y** — vertical location on the qubit grid  
- **t** — time step at which the impact occurred  

---

## 📂 Evaluation

The `evaluation/` directory provides tools for **benchmarking performance** of the detection and clustering pipeline.  

### Metrics
- **Spatial error** — Euclidean distance between predicted and true impact location  
- **Temporal error** — offset between predicted and true peak impact time  
- **Cluster recall** — fraction of events correctly reconstructed  
- **Processing times** — comparisons for CPU vs GPU runs  

### Outputs
- Numerical scorecards  
- LaTeX-ready tables for publication  
- Visualisations of detection accuracy  

---

## 📂 Weights

The `weights/` directory is used to store **YOLOv5 model weights**.  

### Usage
- Place trained `.pt` files here  
- Pretrained weights are not included in the repository due to size restrictions  
- For training instructions or pretrained models, see the [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5)  


## Citations
@software{yolov5,
  title         = {YOLOv5},
  author        = {Jocher, Glenn},
  version       = {7.0}
  year          = {2020},
  url           = {https://github.com/ultralytics/yolov5}
  orcid         = {https://orcid.org/0000-0001-5950-6979}
  doi           = {10.5281/zenodo.3908559}
  date-released = {2020-5-29}
  license       = {AGPL-3.0}
}
