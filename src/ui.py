# =============================================================
# Professional Streamlit UI
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Professional Streamlit desktop-style application for the
AI-based function approximation and tangent line estimation system.

Usage:
    streamlit run src/ui.py
"""

import streamlit as st
import numpy as np
import torch
import os
import sys
import logging
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    set_seed, generate_dataset, create_dataloaders,
    explicit_function, explicit_derivative,
    explicit_derivative_torch, generate_both_datasets
)
from src.model import (
    FunctionApproximator, PINNApproximator, CircleApproximator,
    get_device
)
from src.trainer import Trainer, PINNTrainer
from src.derivatives import AutoDiffCalculator, FiniteDifferenceCalculator, compute_all_derivatives
from src.tangent import TangentLineCalculator, ImplicitTangentCalculator
from src.metrics import PerformanceMetrics
from src.implicit_curve import CircleCurve, CircleTrainer

# Suppress excessive logging in UI
logging.basicConfig(level=logging.WARNING)

# =============================================================
# Page Configuration
# =============================================================

st.set_page_config(
    page_title="AI Tangent Line Approximation System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# Custom CSS
# =============================================================

st.markdown("""
<style>
    /* Main background and font */
    .stApp {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }
    
    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .header-card h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        color: white;
    }
    .header-card p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0.2rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f5f7fa);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1976D2;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        font-size: 0.85rem;
        color: #666;
        margin: 0 0 0.3rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a237e;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a237e;
        border-bottom: 3px solid #1976D2;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #1976D2;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background: #e8f5e9;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #43A047;
        margin: 1rem 0;
    }
    
    /* Equation display */
    .equation-box {
        background: #fff3e0;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #FB8C00;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling removed to support both Light and Dark modes natively */
    
    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #1976D2, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# Cached Functions
# =============================================================

@st.cache_resource
def get_compute_device():
    """Cache the compute device detection."""
    return get_device()


@st.cache_data
def cached_generate_dataset(n_samples, x_min, x_max, noise_std, seed):
    """Cache dataset generation."""
    return generate_dataset(
        func=explicit_function,
        n_samples=n_samples, x_min=x_min, x_max=x_max,
        noise_std=noise_std, seed=seed
    )


@st.cache_resource
def train_model_cached(n_samples, noise_std, epochs, lr, seed, use_pinn, _key):
    """Train and cache the model."""
    device = get_compute_device()
    set_seed(seed)
    
    x, y = cached_generate_dataset(n_samples, -3.0, 3.0, noise_std, seed)
    train_loader, val_loader = create_dataloaders(x, y, batch_size=32, seed=seed)
    
    if use_pinn:
        model = PINNApproximator(hidden_layers=[128, 128, 64, 32], activation='tanh')
        trainer = PINNTrainer(
            model=model, true_derivative_func=explicit_derivative_torch,
            physics_weight=0.1, learning_rate=lr, device=device,
            early_stopping_patience=50
        )
    else:
        model = FunctionApproximator(hidden_layers=[128, 128, 64, 32])
        trainer = Trainer(
            model=model, learning_rate=lr, device=device,
            early_stopping_patience=50
        )
        
    history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=False)
    
    return model, history, device


@st.cache_resource
def train_circle_cached(radius, n_samples, epochs, seed, _key):
    """Train and cache the circle model."""
    device = get_compute_device()
    set_seed(seed)
    
    circle = CircleCurve(radius=radius)
    x_data, y_data = circle.generate_dataset(n_samples=n_samples, seed=seed)
    
    model = CircleApproximator(hidden_layers=[128, 128, 64, 32], radius=radius)
    trainer = CircleTrainer(model, device=device)
    history = trainer.train(x_data, y_data, epochs=epochs, verbose=False)
    
    return model, trainer, circle, history, device


# =============================================================
# Plotting with matplotlib (embedded in Streamlit)
# =============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def create_function_plot(x, y_true, y_pred):
    """Create function approximation plot."""
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, y_true, '#1976D2', linewidth=2.5, label=r'True: $f(x) = x^3 - 2x^2 + \sin(x)$')
    ax1.plot(x, y_pred, '#E53935', linewidth=2.0, linestyle='--', label='Neural Network')
    ax1.fill_between(x, y_true, y_pred, alpha=0.1, color='#E53935')
    ax1.set_title('Neural Network Function Approximation', fontsize=16, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticklabels([])
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    res = y_true - y_pred
    ax2.fill_between(x, res, 0, alpha=0.3, color='#43A047')
    ax2.plot(x, res, '#43A047', linewidth=1.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('Residual', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    mse = float(np.mean(res**2))
    ax2.text(0.97, 0.85, f'MSE = {mse:.2e}', transform=ax2.transAxes, fontsize=10,
             ha='right', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#43A047', alpha=0.9))
    
    fig.tight_layout()
    return fig


def create_tangent_plot(x, y_func, x0, true_tangent, ai_tangent):
    """Create tangent line comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(x, y_func, '#263238', linewidth=2.5, label=r'$f(x)$')
    
    tx = true_tangent.get('x_range')
    if tx is not None and true_tangent.get('y_tangent') is not None:
        ax.plot(tx, true_tangent['y_tangent'], '#1976D2', linewidth=2.5,
                label=f"True: {true_tangent['equation']}")
    if tx is not None and ai_tangent.get('y_tangent') is not None:
        ax.plot(tx, ai_tangent['y_tangent'], '#E53935', linewidth=2.5,
                linestyle='--', label=f"AI: {ai_tangent['equation']}")
    
    ax.scatter([x0], [true_tangent['f_x0']], color='#FB8C00', s=150,
               zorder=5, edgecolors='black', linewidths=1.5,
               label=f'x₀ = {x0}')
    
    ax.set_title('Tangent Line: True vs AI-Derived', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig


def create_loss_plot(history):
    """Create training loss plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.semilogy(epochs, history['train_loss'], '#1976D2', linewidth=2, label='Train Loss')
    ax.semilogy(epochs, history['val_loss'], '#E53935', linewidth=2, label='Val Loss')
    
    best_ep = int(np.argmin(history['val_loss'])) + 1
    best_val = min(history['val_loss'])
    ax.axvline(best_ep, color='#43A047', linestyle=':', alpha=0.7)
    ax.scatter([best_ep], [best_val], color='#43A047', s=100, zorder=5,
               edgecolors='black', label=f'Best (ep {best_ep})')
    
    ax.set_title('Training Progress', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss (log)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig


def create_derivative_plot(x, true_d, ad_d, fd_d):
    """Create derivative comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, true_d, '#263238', linewidth=2.5, label=r"True: $f'(x)$")
    ax.plot(x, ad_d, '#1976D2', linewidth=2, linestyle='--', label='AutoDiff')
    ax.plot(x, fd_d, '#E53935', linewidth=2, linestyle=':', label='Finite Diff')
    ax.set_title('Derivative Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel("f'(x)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig


def create_circle_plot(circle, x0, y0, true_tangent, ai_tangent, nn_x, nn_y):
    """Create circle visualization plot."""
    fig, ax = plt.subplots(figsize=(9, 9))
    
    theta = np.linspace(0, 2*np.pi, 500)
    cx = circle.radius * np.cos(theta)
    cy = circle.radius * np.sin(theta)
    ax.plot(cx, cy, '#263238', linewidth=2.5, label=f'x² + y² = {circle.radius**2:.0f}')
    
    if nn_x is not None:
        ax.plot(nn_x, nn_y, '#1976D2', linewidth=2, linestyle='--', label='NN Approx')
    
    tx = true_tangent.get('x_range')
    if tx is not None and true_tangent.get('y_tangent') is not None:
        ax.plot(tx, true_tangent['y_tangent'], '#43A047', linewidth=2.5, label='True Tangent')
    if tx is not None and ai_tangent.get('y_tangent') is not None:
        ax.plot(tx, ai_tangent['y_tangent'], '#E53935', linewidth=2.5, linestyle='--', label='AI Tangent')
    
    ax.scatter([x0], [y0], color='#FB8C00', s=150, zorder=5, edgecolors='black', linewidths=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Circle with Tangent at ({x0}, {y0:.2f})', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3, linestyle='--')
    r = circle.radius
    ax.set_xlim(-r-2, r+2)
    ax.set_ylim(-r-2, r+2)
    fig.tight_layout()
    return fig


# =============================================================
# Sidebar
# =============================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    page = st.radio(
        "📑 Navigation",
        ["🏠 Dashboard", "📈 Function Approx", "📐 Tangent Lines",
         "🔬 Derivatives", "⭕ Circle / Conic", "📊 Metrics"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Parameters")
    
    n_samples = st.slider("Training Samples", 200, 2000, 1000, 100)
    epochs = st.slider("Max Epochs", 100, 2000, 1000, 100)
    lr = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    noise_std = st.slider("Noise Std", 0.0, 0.2, 0.0, 0.01)
    x0_val = st.slider("x₀ (tangent point)", -2.5, 2.5, 1.5, 0.1)
    
    st.markdown("---")
    use_pinn = st.checkbox("Enable Physics-Informed NN (PINN) Bonus", value=False)
    
    st.markdown("---")
    st.markdown("### 📋 Project Info")
    st.markdown("""
    **Student:** Mujeeb-Ur-Rehman Sahito  
    **Roll No:** 25-BSCS-43  
    **University:** Sheikh Ayaz University  
    **Dept:** Computer Science
    """)


# =============================================================
# Header
# =============================================================

st.markdown("""
<div class="header-card">
    <h1>🧠 AI-Based Tangent Line Approximation System</h1>
    <p>Neural Network Function Approximation with Automatic Differentiation</p>
    <p style="opacity:0.7; font-size:0.9rem;">
        By Mujeeb-Ur-Rehman Sahito (25-BSCS-43) · Sheikh Ayaz University · Computer Science Department
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================
# Train Model (cached)
# =============================================================

cache_key = f"{n_samples}_{noise_std}_{epochs}_{lr}_{use_pinn}"

with st.spinner("🔄 Training neural network... This may take a moment."):
    model, history, device = train_model_cached(n_samples, noise_std, epochs, lr, 42, use_pinn, cache_key)

# Get predictions
model.eval()
x_data, y_data = cached_generate_dataset(n_samples, -3.0, 3.0, noise_std, 42)
x_clean, y_clean = cached_generate_dataset(n_samples, -3.0, 3.0, 0.0, 42)

with torch.no_grad():
    x_t = torch.FloatTensor(x_data.reshape(-1, 1)).to(device)
    y_pred = model(x_t).cpu().numpy().flatten()
    x_t_clean = torch.FloatTensor(x_clean.reshape(-1, 1)).to(device)
    y_pred_clean = model(x_t_clean).cpu().numpy().flatten()


# =============================================================
# Pages
# =============================================================

if page == "🏠 Dashboard":
    st.markdown('<div class="section-header">📊 System Overview</div>', unsafe_allow_html=True)
    
    # Key metrics
    mse_val = float(np.mean((y_clean - y_pred_clean)**2))
    r2_val = PerformanceMetrics.r_squared(y_clean, y_pred_clean)
    best_loss = min(history['val_loss'])
    total_epochs = len(history['train_loss'])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Function MSE", f"{mse_val:.2e}")
    with c2:
        st.metric("R² Score", f"{r2_val:.6f}")
    with c3:
        st.metric("Best Val Loss", f"{best_loss:.2e}")
    with c4:
        st.metric("Epochs Run", f"{total_epochs}")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📈 Function Approximation")
        fig = create_function_plot(x_clean, y_clean, y_pred_clean)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("#### 📉 Training Progress")
        fig = create_loss_plot(history)
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🧪 Project Functions:</strong><br>
        • <strong>Explicit:</strong> f(x) = x³ − 2x² + sin(x) on [−3, 3]<br>
        • <strong>Implicit:</strong> x² + y² = 25 (circle, radius 5)<br>
        • <strong>Architecture:</strong> Input(1) → 128 → 128 → 64 → 32 → Output(1) with ReLU
    </div>
    """, unsafe_allow_html=True)

elif page == "📈 Function Approx":
    st.markdown('<div class="section-header">📈 Function Approximation</div>', unsafe_allow_html=True)
    
    fig = create_function_plot(x_clean, y_clean, y_pred_clean)
    st.pyplot(fig)
    plt.close(fig)
    
    mse_val = float(np.mean((y_clean - y_pred_clean)**2))
    mae_val = float(np.mean(np.abs(y_clean - y_pred_clean)))
    r2_val = PerformanceMetrics.r_squared(y_clean, y_pred_clean)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mse_val:.6e}")
    c2.metric("MAE", f"{mae_val:.6e}")
    c3.metric("R²", f"{r2_val:.6f}")
    
    st.markdown("#### Training Curves")
    fig = create_loss_plot(history)
    st.pyplot(fig)
    plt.close(fig)

elif page == "📐 Tangent Lines":
    st.markdown('<div class="section-header">📐 Tangent Line Comparison</div>', unsafe_allow_html=True)
    
    x_range = np.linspace(x0_val - 2, x0_val + 2, 200).astype(np.float32)
    
    true_tangent = TangentLineCalculator.compute_true_tangent(
        x0_val, explicit_function, explicit_derivative, x_range)
    ai_tangent = TangentLineCalculator.compute_ai_tangent(
        x0_val, model, device, x_range)
    comparison = TangentLineCalculator.compare_tangent_lines(true_tangent, ai_tangent)
    
    fig = create_tangent_plot(x_clean, y_clean, x0_val, true_tangent, ai_tangent)
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown(f"""
    <div class="equation-box">
        <strong>True:</strong> {true_tangent['equation_point_form']}<br>
        <strong>AI:</strong>&nbsp;&nbsp;&nbsp; {ai_tangent['equation_point_form']}
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Slope Error", f"{comparison['slope_error']:.6e}")
    c2.metric("Intercept Error", f"{comparison['intercept_error']:.6e}")
    if 'tangent_mse' in comparison:
        c3.metric("Tangent MSE", f"{comparison['tangent_mse']:.6e}")

elif page == "🔬 Derivatives":
    st.markdown('<div class="section-header">🔬 Derivative Analysis</div>', unsafe_allow_html=True)
    
    deriv_results = compute_all_derivatives(model, x_clean, explicit_function, explicit_derivative, device)
    
    fig = create_derivative_plot(
        x_clean, deriv_results['true_derivative'],
        deriv_results['autograd_derivative'],
        deriv_results['finite_diff_derivative']
    )
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("#### Error Metrics")
    ad_mse = float(np.mean((deriv_results['true_derivative'] - deriv_results['autograd_derivative'])**2))
    fd_mse = float(np.mean((deriv_results['true_derivative'] - deriv_results['finite_diff_derivative'])**2))
    
    c1, c2 = st.columns(2)
    c1.metric("AutoDiff MSE vs True", f"{ad_mse:.6e}")
    c2.metric("Finite Diff MSE vs True", f"{fd_mse:.6e}")
    
    st.markdown("""
    <div class="info-box">
        <strong>Methods:</strong><br>
        • <strong>AutoDiff:</strong> PyTorch autograd (exact for NN)<br>
        • <strong>Finite Difference:</strong> Central difference [f(x+h)−f(x−h)]/(2h)
    </div>
    """, unsafe_allow_html=True)

elif page == "⭕ Circle / Conic":
    st.markdown('<div class="section-header">⭕ Implicit Circle: x² + y² = 25</div>', unsafe_allow_html=True)
    
    circle_x0 = st.slider("Circle tangent x₀", -4.5, 4.5, 3.0, 0.1, key="cx0")
    
    circle_key = f"circle_{n_samples}_{epochs}"
    with st.spinner("Training circle model..."):
        c_model, c_trainer, circle, c_history, c_device = train_circle_cached(5.0, 500, 500, 42, circle_key)
    
    c_y0 = float(np.sqrt(25 - min(circle_x0**2, 24.99)))
    tangent_xr = np.linspace(circle_x0 - 3, circle_x0 + 3, 200).astype(np.float32)
    
    true_ct = circle.tangent_at_point(circle_x0, c_y0, tangent_xr)
    ai_ct = c_trainer.compute_ai_tangent(circle_x0, tangent_xr)
    
    x_plot = np.linspace(-4.99, 4.99, 500).astype(np.float32)
    nn_y = c_trainer.predict(x_plot)
    
    fig = create_circle_plot(circle, circle_x0, c_y0, true_ct, ai_ct, x_plot, nn_y)
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown(f"""
    <div class="equation-box">
        <strong>Point:</strong> ({circle_x0:.2f}, {c_y0:.4f})<br>
        <strong>dy/dx = −x/y = {-circle_x0/c_y0:.4f}</strong><br>
        <strong>True:</strong> {true_ct['equation_point_form']}<br>
        <strong>AI:</strong>&nbsp;&nbsp;&nbsp; {ai_ct['equation_point_form']}
    </div>
    """, unsafe_allow_html=True)
    
    slope_err = abs(true_ct['slope'] - ai_ct['slope'])
    y0_err = abs(c_y0 - ai_ct['y0'])
    c1, c2 = st.columns(2)
    c1.metric("Slope Error", f"{slope_err:.6e}")
    c2.metric("y₀ Error", f"{y0_err:.6e}")

elif page == "📊 Metrics":
    st.markdown('<div class="section-header">📊 Complete Metrics Dashboard</div>', unsafe_allow_html=True)
    
    # Compute all metrics
    x_range = np.linspace(x0_val - 2, x0_val + 2, 200).astype(np.float32)
    true_t = TangentLineCalculator.compute_true_tangent(x0_val, explicit_function, explicit_derivative, x_range)
    ai_t = TangentLineCalculator.compute_ai_tangent(x0_val, model, device, x_range)
    
    ad = AutoDiffCalculator(model, device)
    _, ad_d = ad.compute_derivative(x_clean)
    true_d = explicit_derivative(x_clean)
    
    metrics = PerformanceMetrics.compute_all_metrics(
        y_clean, y_pred_clean, true_t, ai_t, true_d, ad_d)
    
    st.markdown("#### Function Approximation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE", f"{metrics['function_mse']:.6e}")
    c2.metric("MAE", f"{metrics['function_mae']:.6e}")
    c3.metric("RMSE", f"{metrics['function_rmse']:.6e}")
    c4.metric("R²", f"{metrics['function_r_squared']:.6f}")
    
    st.markdown("#### Tangent Line")
    c1, c2, c3 = st.columns(3)
    c1.metric("Slope Error", f"{metrics['slope_error']:.6e}")
    c2.metric("Intercept Error", f"{metrics['intercept_error']:.6e}")
    if 'tangent_line_mse' in metrics:
        c3.metric("Tangent MSE", f"{metrics['tangent_line_mse']:.6e}")
    
    st.markdown("#### Derivative Estimation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deriv MSE", f"{metrics['derivative_mse']:.6e}")
    c2.metric("Deriv MAE", f"{metrics['derivative_mae']:.6e}")
    c3.metric("Deriv RMSE", f"{metrics['derivative_rmse']:.6e}")
    c4.metric("Deriv R²", f"{metrics['derivative_r_squared']:.6f}")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("#### 📋 Full Metrics Table")
    import pandas as pd
    name_map = {
        'function_mse': 'Function MSE', 'function_mae': 'Function MAE',
        'function_rmse': 'Function RMSE', 'function_r_squared': 'Function R²',
        'function_max_error': 'Function Max Error', 'slope_error': 'Slope Error',
        'intercept_error': 'Intercept Error', 'tangent_line_mse': 'Tangent Line MSE',
        'derivative_mse': 'Derivative MSE', 'derivative_mae': 'Derivative MAE',
        'derivative_rmse': 'Derivative RMSE', 'derivative_r_squared': 'Derivative R²',
        'derivative_max_error': 'Derivative Max Error',
    }
    rows = []
    for k, v in metrics.items():
        name = name_map.get(k, k.replace('_', ' ').title())
        rows.append({'Metric': name, 'Value': f"{v:.6e}" if isinstance(v, float) else str(v)})
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================
# Footer
# =============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p style="font-size: 0.9rem;">
        AI-Based Neural Network System for Function Approximation · 
        Built with PyTorch & Streamlit · 
        © 2026 Mujeeb-Ur-Rehman Sahito
    </p>
</div>
""", unsafe_allow_html=True)
