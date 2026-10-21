# Real-Time Road Hazard Detection

A deep learning system that detects road users and road hazards in video, estimates risk, and warns the driver in real time.

> Semester project for the course **Methods and Applications of Artificial Intelligence**.

```
🚶 Pedestrian   ~8 m    🔴 HIGH RISK
🚗 Vehicle      ~22 m   🟡 MEDIUM
⚠️ Pothole      ~12 m   🟡 MEDIUM
```

---

## Table of Contents

- [About](#about)
- [System Architecture](#system-architecture)
- [Detected Classes](#detected-classes)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Roadmap](#roadmap)
- [Team](#team)

---

## About

The goal is not just object detection, but a complete system that could be used in real driving:

**camera / video → detection → tracking → distance estimation → risk assessment → warning**

Each module is developed and evaluated separately, then connected into a single pipeline.

## System Architecture

| # | Module | Description | Technology |
|---|---|---|---|
| 1 | Detection | Detects objects in every frame (two models: road users + road damage and roadwork) | YOLOv8 / YOLO11 (Ultralytics) |
| 2 | Tracking | Assigns IDs to objects across frames, tracks trajectory and approach speed | ByteTrack / BoT-SORT |
| 3 | Distance | Estimates distance from a single camera | Pinhole camera model, optionally Depth Anything V2 |
| 4 | Risk assessment | Combines class, distance, lane position and time-to-collision (TTC) | Rule-based |
| 5 | Warning | Video overlay, audio alert, risk level LOW / MEDIUM / HIGH | OpenCV, Streamlit / Gradio |

**Risk score:**

```
risk = class_weight × distance_factor × position_factor × approach_factor
```

## Detected Classes

**MVP:**
- vehicle (car, truck, bus, motorcycle)
- pedestrian
- cyclist
- traffic sign
- pothole

**Planned extensions:**
- roadwork (cones, barriers)
- road surface damage (cracks)
- traffic sign type classification

## Datasets

| Dataset | Purpose |
|---|---|
| [BDD100K](https://www.vis.xyz/bdd100k/) | Vehicles, pedestrians, cyclists, traffic signs and lights |
| [RDD2022](https://github.com/sekilab/RoadDamageDetector) | Road surface damage and potholes |
| [Roboflow Universe](https://universe.roboflow.com/) | Additional datasets for potholes, cones and roadwork |
| [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) | Traffic sign classification (optional) |
| Own recordings | Local test set recorded on roads in Bosnia and Herzegovina |


## Repository Structure

```
road-hazard-detection/
├── data/                 # dataset download and conversion scripts (data not in git)
├── notebooks/            # EDA, experiments, training on Colab/Kaggle
├── src/
│   ├── detection/        # YOLO training and inference
│   ├── tracking/         # object tracking
│   ├── distance/         # camera calibration and distance estimation
│   ├── risk/             # risk assessment logic
│   ├── alert/            # overlay and audio warnings
│   └── pipeline.py       # connects all modules
├── app/                  # demo application (Streamlit / Gradio)
├── configs/              # dataset and model configurations
├── models/               # trained weights (or links to them)
├── results/              # metrics, plots, example outputs
├── docs/                 # report and presentation
├── requirements.txt
└── README.md
```

## Installation

> 🚧 **Work in progress.** The project is in its early stages; setup instructions will be finalized once the first working version is available.

Planned setup:
```bash
git clone https://github.com/<username>/road-hazard-detection.git
cd road-hazard-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Main dependencies:** Python 3.10+, PyTorch, Ultralytics, OpenCV, NumPy, Streamlit

## Usage

> The commands below are planned and will be updated as the project progresses.

**Process a video:**
```bash
python src/pipeline.py --source video.mp4 --output results/output.mp4
```

**Live camera:**
```bash
python src/pipeline.py --source 0
```

**Demo application:**
```bash
streamlit run app/app.py
```

## Evaluation

| What we measure | Metric |
|---|---|
| Detection | mAP@0.5, mAP@0.5:0.95, per-class precision and recall, confusion matrix |
| Speed | FPS on GPU and CPU |
| Distance | Mean absolute error (m) on recordings with measured distances |
| Risk | Recall for HIGH-risk situations, number of false alarms |

Results will be published in the `results/` folder.

## Roadmap

- [ ] Week 1: dataset selection, environment setup
- [ ] Week 2: data preparation and EDA
- [ ] Weeks 3–5: training detection models
- [ ] Week 6: tracking and distance estimation
- [ ] Week 7: risk assessment and warnings — **MVP**
- [ ] Weeks 8–9: experiments and ablations
- [ ] Week 10: demo application and evaluation
- [ ] Weeks 11–12: report, presentation and submission

## Team

| Member |
|---|
| Denan Poturak |
| Hamza Hrnjić |
| Safet Čomor |
| Emir Duvnjak |
