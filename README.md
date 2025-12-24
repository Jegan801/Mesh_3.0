# Mesh Project 2.0

Mesh Project 2.0 is a CAE (Computer-Aided Engineering) mesh quality analysis and AI-driven risk assessment tool.  
It analyzes finite element meshes, detects mesh quality issues, and generates hybrid risk scores using rule-based logic and machine learning models.

This project is designed for mesh validation, debugging, and comparison workflows in automotive, aerospace, and structural CAE pipelines.

---

## Features

- Load mesh data from CSV files (nodes and elements)
- Compute geometric quality metrics
  - Area
  - Aspect ratio
  - Edge length ratios
- Rule-based mesh error detection
- Hybrid AI risk scoring (rules + ML)
- Interactive 3D visualization using Plotly
- Side-by-side mesh comparison and error debugging
- Modular and scalable architecture

---

## Project Structure
```
Mesh-project-2.0/
├── main.py # Main entry point
├── core/ # Core mesh handling
│ ├── mesh_loader.py # Load mesh from CSV
│ ├── mesh_neighbors.py # Build element neighbors
│ └── mesh_objects.py # Mesh data structures
├── quality/ # Mesh quality analysis
│ ├── metrics.py # Geometric quality metrics
│ └── rules.py # Rule-based error detection
├── ai/ # AI and risk modeling
│ ├── feature_builder.py # Feature engineering
│ ├── risk_model.py # Rule-based risk scoring
│ ├── rf_model.py # Random Forest model
│ └── hybrid_risk.py # Hybrid risk computation
├── analysis/ # Analysis utilities
│ ├── compare_meshes.py # Mesh comparison logic
│ └── scorecard.py # Reporting and scorecards
├── visualization/ # 3D visualization
│ ├── hybrid_comparison_3d.py # Side-by-side comparison
│ ├── mesh_error_debug_3d.py # Error visualization
│ └── plot_utils.py
├── data/ # Mesh datasets (git-ignored)
│ ├── cad/
│ ├── first_mesh/
│ └── final_mesh/
├── requirements.txt # Python dependencies
└── README.md
```
yaml
Copy code

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Obsilisk/Mesh-project-2.0.git
cd Mesh-project-2.0
Create a Virtual Environment
Windows


python -m venv .venv
.venv\Scripts\activate
Linux / macOS


python3 -m venv .venv
source .venv/bin/activate
Install Dependencies
```
```
pip install -r requirements.txt
```
Usage
Run the full analysis pipeline:

```
python main.py
```
Pipeline Steps
Load first and final meshes

Compute quality metrics

Detect mesh errors

Compute hybrid AI risk scores

Generate interactive 3D visualizations

Print high-risk elements in the console

Customization
Edit main.py to:

Change mesh input files

Adjust quality thresholds

Enable or disable ML scoring

Extend:

quality/rules.py for new mesh rules

ai/feature_builder.py for additional features

visualization/ for new plots

Output
```
first_vs_final_mesh_comparison.html
```
Interactive side-by-side 3D mesh comparison
```
first_mesh_error_debug.html
```
Detailed error visualization with highlighted elements

Console output:

High-risk elements

Rule violations

Risk scores

Dependencies
```
numpy

pandas

matplotlib

plotly

scikit-learn
```

Contributing
Fork the repository

Create a feature branch

Commit your changes

Open a pull request

License
This project is licensed under the MIT License.
See the LICENSE file for details.

# SRM Mesh Agents

## Project Description
[Add a brief description of the project here.]

## Ignored Directories
The following directories are ignored by Git to avoid committing large or sensitive data:
- `data/`: General data files.
- `data_2/`: Raw mesh data.
- `PART_02/`: Specific project data.
- `data_3/`: Additional datasets.
- `experiments/`: Experiment outputs and results.
- `models/`: Trained models.
- `outputs/`: Output files and reports.
- `.venv/`, `venv/`, `env/`, `ENV/`: Virtual environments.

## Installation
[Add installation instructions here.]

## Usage
[Add usage instructions here.]

## Contributing
[Add contributing guidelines here.]



