# 🧠 AI-Based Neural Network System for Function Approximation, Derivative Computation, and Tangent Line Estimation

> **Using Automatic Differentiation with PyTorch**

---

## 📋 Project Information

| Field | Details |
|-------|---------|
| **Student** | Mujeeb-Ur-Rehman Sahito |
| **Roll No** | 25-BSCS-43 |
| **University** | Sheikh Ayaz University |
| **Department** | Computer Science Department |

---

## 📖 Overview

This project implements an AI-based system that uses deep neural networks to:

1. **Approximate mathematical functions** using feedforward neural networks
2. **Compute derivatives** using automatic differentiation (PyTorch autograd) and finite difference methods
3. **Estimate tangent lines** at specified points and compare with true analytic tangent lines
4. **Handle implicit functions** (conic sections like circles) with implicit differentiation
5. **Evaluate performance** using comprehensive metrics (MSE, MAE, RMSE, R², etc.)
6. **Visualize results** with publication-quality graphs

### Functions Studied

- **Explicit**: f(x) = x³ − 2x² + sin(x) on [-3, 3]
- **Implicit**: x² + y² = 25 (circle, radius 5)

---

## 🏗️ Project Structure

```
project_root/
├── data/                    # Generated datasets
├── models/                  # Saved model checkpoints
├── outputs/                 # Program outputs and logs
├── figures/                 # Generated plots (PNG + PDF)
├── reports/                 # Academic report
├── notebooks/               # Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Synthetic dataset generation
│   ├── model.py             # Neural network architecture
│   ├── trainer.py           # Training loop with early stopping
│   ├── derivatives.py       # Automatic & finite difference derivatives
│   ├── tangent.py           # Tangent line computation
│   ├── metrics.py           # Performance metrics
│   ├── visualization.py     # Professional plotting
│   ├── implicit_curve.py    # Conic sections / implicit functions
│   └── ui.py                # Streamlit UI application
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── PROJECT_PROGRESS.md      # Progress tracker
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Main Script

```bash
python main.py
```

This will:
- Generate synthetic datasets
- Train neural networks
- Compute derivatives and tangent lines
- Generate all visualizations
- Display metrics

### 3. Launch the UI

```bash
streamlit run src/ui.py
```

### 4. Open the Notebook

```bash
jupyter notebook notebooks/project_notebook.ipynb
```

---

## 🧪 Neural Network Architecture

```
Input (1) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output (1)
```

- **Optimizer**: Adam with learning rate scheduling
- **Loss**: Mean Squared Error (MSE)
- **Initialization**: Xavier/Glorot
- **Regularization**: Early stopping with patience
- **Training**: 1000 epochs, batch size 32

---

## 📊 Metrics Computed

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error of function approximation |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination |
| Slope Error | Difference between AI and true tangent slopes |
| Intercept Error | Difference between AI and true intercepts |
| Tangent MSE | MSE between AI and true tangent lines |
| Derivative Error | Error in derivative estimation |

---

## 📈 Visualizations

All figures are saved in `figures/` in both PNG (300 DPI) and PDF formats:

1. Function approximation comparison
2. Tangent line comparison
3. Training loss curve
4. Derivative comparison
5. Error analysis
6. Circle/conic visualization
7. Implicit tangent line
8. Noise comparison

---

## 📝 License

This project is submitted as part of academic coursework at Sheikh Ayaz University.

---

*Built with PyTorch, Matplotlib, Seaborn, and Streamlit*
