# =============================================================
# Main Execution Script
# AI-Based Neural Network System for Function Approximation,
# Derivative Computation, and Tangent Line Estimation
#
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# University: Sheikh Ayaz University
# Department: Computer Science Department
# =============================================================

"""
Main entry point for the AI-based function approximation system.

Executes the complete pipeline:
1. Dataset generation (clean + noisy)
2. Neural network training (standard + PINN)
3. Derivative computation (autograd + finite difference)
4. Tangent line estimation and comparison
5. Performance metrics evaluation
6. Professional visualization generation
7. Implicit circle experiment
8. Results export

Usage:
    python main.py
"""

import os
import sys
import time
import logging
import numpy as np
import torch

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ---- Setup Logging ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/run.log', mode='w')
    ]
)
logger = logging.getLogger('main')

# ---- Project Imports ----
from src.dataset import (
    set_seed, generate_dataset, create_dataloaders,
    save_dataset, explicit_function, explicit_derivative,
    explicit_derivative_torch, generate_both_datasets
)
from src.model import (
    FunctionApproximator, PINNApproximator, CircleApproximator,
    get_device, save_model
)
from src.trainer import Trainer, PINNTrainer
from src.derivatives import (
    AutoDiffCalculator, FiniteDifferenceCalculator,
    compute_all_derivatives, compare_derivatives
)
from src.tangent import TangentLineCalculator, ImplicitTangentCalculator
from src.metrics import PerformanceMetrics
from src.visualization import Visualizer
from src.implicit_curve import run_circle_experiment


# =============================================================
# Configuration
# =============================================================

CONFIG = {
    # Reproducibility
    'seed': 42,
    
    # Dataset
    'x_min': -3.0,
    'x_max': 3.0,
    'n_samples': 1000,
    'noise_std': 0.05,
    'batch_size': 32,
    'val_split': 0.2,
    
    # Model
    'hidden_layers': [128, 128, 64, 32],
    'activation': 'relu',
    
    # Training
    'epochs': 1000,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'early_stopping_patience': 50,
    'scheduler_type': 'reduce_on_plateau',
    
    # Tangent
    'x0': 1.5,
    
    # Circle
    'circle_radius': 5.0,
    'circle_x0': 3.0,
    'circle_epochs': 500,
    'circle_n_samples': 500,
    
    # PINN
    'pinn_physics_weight': 0.1,
    'pinn_epochs': 1000,
    
    # Output
    'figures_dir': 'figures',
    'models_dir': 'models',
    'data_dir': 'data',
    'outputs_dir': 'outputs',
}


def create_directories() -> None:
    """Create all necessary output directories."""
    for dir_name in ['figures', 'models', 'data', 'outputs', 'reports', 'notebooks']:
        os.makedirs(dir_name, exist_ok=True)
    logger.info("Output directories created.")


# =============================================================
# Phase 1: Dataset Generation
# =============================================================

def phase_dataset_generation(config: dict) -> dict:
    """Generate clean and noisy datasets."""
    logger.info("=" * 60)
    logger.info("PHASE 1: DATASET GENERATION")
    logger.info("=" * 60)
    
    set_seed(config['seed'])
    
    # Generate both clean and noisy datasets
    datasets = generate_both_datasets(
        x_min=config['x_min'],
        x_max=config['x_max'],
        n_samples=config['n_samples'],
        noise_std=config['noise_std'],
        seed=config['seed']
    )
    
    x_clean, y_clean = datasets['clean']
    x_noisy, y_noisy = datasets['noisy']
    
    # Save datasets
    save_dataset(x_clean, y_clean, os.path.join(config['data_dir'], 'dataset_clean'))
    save_dataset(x_noisy, y_noisy, os.path.join(config['data_dir'], 'dataset_noisy'))
    
    # Create DataLoaders for clean data
    train_loader, val_loader = create_dataloaders(
        x_clean, y_clean,
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        seed=config['seed']
    )
    
    # Create DataLoaders for noisy data
    train_loader_noisy, val_loader_noisy = create_dataloaders(
        x_noisy, y_noisy,
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        seed=config['seed']
    )
    
    logger.info(f"Clean dataset: {len(x_clean)} samples")
    logger.info(f"Noisy dataset: {len(x_noisy)} samples (noise_std={config['noise_std']})")
    
    return {
        'x_clean': x_clean, 'y_clean': y_clean,
        'x_noisy': x_noisy, 'y_noisy': y_noisy,
        'train_loader': train_loader, 'val_loader': val_loader,
        'train_loader_noisy': train_loader_noisy,
        'val_loader_noisy': val_loader_noisy,
    }


# =============================================================
# Phase 2: Model Training
# =============================================================

def phase_model_training(config: dict, data: dict, device: torch.device) -> dict:
    """Train neural network models (standard + noisy + PINN)."""
    logger.info("=" * 60)
    logger.info("PHASE 2: NEURAL NETWORK TRAINING")
    logger.info("=" * 60)
    
    results = {}
    
    # --- Standard NN (clean data) ---
    logger.info("-" * 40)
    logger.info("Training Standard NN on clean data...")
    logger.info("-" * 40)
    
    set_seed(config['seed'])
    model = FunctionApproximator(
        hidden_layers=config['hidden_layers'],
        activation=config['activation']
    )
    print(model.get_architecture_summary())
    
    trainer = Trainer(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device,
        scheduler_type=config['scheduler_type'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    history = trainer.fit(
        data['train_loader'], data['val_loader'],
        epochs=config['epochs'],
        verbose=True,
        log_interval=100
    )
    
    save_model(model, os.path.join(config['models_dir'], 'model_clean.pt'),
               metadata={'type': 'clean', 'best_val_loss': min(history['val_loss'])})
    
    results['model'] = model
    results['history'] = history
    results['trainer'] = trainer
    
    # --- Standard NN (noisy data) ---
    logger.info("-" * 40)
    logger.info("Training Standard NN on noisy data...")
    logger.info("-" * 40)
    
    set_seed(config['seed'])
    model_noisy = FunctionApproximator(
        hidden_layers=config['hidden_layers'],
        activation=config['activation']
    )
    
    trainer_noisy = Trainer(
        model=model_noisy,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device,
        scheduler_type=config['scheduler_type'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    history_noisy = trainer_noisy.fit(
        data['train_loader_noisy'], data['val_loader_noisy'],
        epochs=config['epochs'],
        verbose=True,
        log_interval=100
    )
    
    save_model(model_noisy, os.path.join(config['models_dir'], 'model_noisy.pt'),
               metadata={'type': 'noisy', 'best_val_loss': min(history_noisy['val_loss'])})
    
    results['model_noisy'] = model_noisy
    results['history_noisy'] = history_noisy
    
    # --- PINN ---
    logger.info("-" * 40)
    logger.info("Training Physics-Informed Neural Network...")
    logger.info("-" * 40)
    
    set_seed(config['seed'])
    model_pinn = PINNApproximator(
        hidden_layers=config['hidden_layers'],
        activation='tanh'
    )
    
    pinn_trainer = PINNTrainer(
        model=model_pinn,
        true_derivative_func=explicit_derivative_torch,
        physics_weight=config['pinn_physics_weight'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device,
        scheduler_type=config['scheduler_type'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    history_pinn = pinn_trainer.fit(
        data['train_loader'], data['val_loader'],
        epochs=config['pinn_epochs'],
        verbose=True,
        log_interval=100
    )
    
    save_model(model_pinn, os.path.join(config['models_dir'], 'model_pinn.pt'),
               metadata={'type': 'pinn', 'best_val_loss': min(history_pinn['val_loss'])})
    
    results['model_pinn'] = model_pinn
    results['history_pinn'] = history_pinn
    
    return results


# =============================================================
# Phase 3: Derivatives
# =============================================================

def phase_derivatives(config: dict, data: dict, model: torch.nn.Module,
                      device: torch.device) -> dict:
    """Compute derivatives using all methods."""
    logger.info("=" * 60)
    logger.info("PHASE 3: DERIVATIVE COMPUTATION")
    logger.info("=" * 60)
    
    x = data['x_clean']
    
    # Compute all derivatives
    deriv_results = compute_all_derivatives(
        model=model, x=x,
        true_func=explicit_function,
        true_deriv_func=explicit_derivative,
        device=device
    )
    
    # Compare
    comparison = compare_derivatives(
        x=x,
        true_derivative=deriv_results['true_derivative'],
        auto_diff_derivative=deriv_results['autograd_derivative'],
        finite_diff_derivative=deriv_results['finite_diff_derivative']
    )
    
    # Print comparison
    logger.info("\nDerivative Comparison Results:")
    for method, metrics in comparison.items():
        logger.info(f"  {method}:")
        for k, v in metrics.items():
            logger.info(f"    {k}: {v:.6e}")
    
    deriv_results['comparison'] = comparison
    return deriv_results


# =============================================================
# Phase 4: Tangent Lines
# =============================================================

def phase_tangent_lines(config: dict, data: dict, model: torch.nn.Module,
                        device: torch.device) -> dict:
    """Compute and compare tangent lines."""
    logger.info("=" * 60)
    logger.info("PHASE 4: TANGENT LINE COMPUTATION")
    logger.info("=" * 60)
    
    x0 = config['x0']
    x_range = np.linspace(x0 - 2, x0 + 2, 200).astype(np.float32)
    
    # True tangent
    true_tangent = TangentLineCalculator.compute_true_tangent(
        x0=x0,
        func=explicit_function,
        derivative_func=explicit_derivative,
        x_range=x_range
    )
    
    # AI tangent
    ai_tangent = TangentLineCalculator.compute_ai_tangent(
        x0=x0,
        model=model,
        device=device,
        x_range=x_range
    )
    
    # Compare
    comparison = TangentLineCalculator.compare_tangent_lines(true_tangent, ai_tangent)
    
    # Display
    table = PerformanceMetrics.format_comparison_table(true_tangent, ai_tangent, comparison)
    print(table)
    logger.info(table)
    
    return {
        'true_tangent': true_tangent,
        'ai_tangent': ai_tangent,
        'comparison': comparison,
        'x0': x0,
    }


# =============================================================
# Phase 5: Metrics
# =============================================================

def phase_metrics(data: dict, model: torch.nn.Module, device: torch.device,
                  tangent_results: dict, deriv_results: dict) -> dict:
    """Compute comprehensive performance metrics."""
    logger.info("=" * 60)
    logger.info("PHASE 5: PERFORMANCE METRICS")
    logger.info("=" * 60)
    
    x = data['x_clean']
    y_true = data['y_clean']
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(device)
        y_pred = model(x_tensor).cpu().numpy().flatten()
    
    # Compute all metrics
    metrics = PerformanceMetrics.compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        true_tangent=tangent_results['true_tangent'],
        ai_tangent=tangent_results['ai_tangent'],
        true_deriv=deriv_results['true_derivative'],
        ai_deriv=deriv_results['autograd_derivative']
    )
    
    # Display
    table = PerformanceMetrics.format_metrics_table(metrics)
    print(table)
    logger.info(table)
    
    # Save
    PerformanceMetrics.save_metrics(metrics, 'outputs/metrics.json')
    
    return {'metrics': metrics, 'y_pred': y_pred}


# =============================================================
# Phase 6: Visualization
# =============================================================

def phase_visualization(config: dict, data: dict, model: torch.nn.Module,
                        model_noisy: torch.nn.Module, model_pinn: torch.nn.Module,
                        device: torch.device, training_results: dict,
                        deriv_results: dict, tangent_results: dict,
                        metrics_results: dict) -> None:
    """Generate all professional visualizations."""
    logger.info("=" * 60)
    logger.info("PHASE 6: VISUALIZATION")
    logger.info("=" * 60)
    
    viz = Visualizer(output_dir=config['figures_dir'])
    x = data['x_clean']
    y_true = data['y_clean']
    y_pred = metrics_results['y_pred']
    
    # 1. Function Approximation
    viz.plot_function_approximation(x, y_true, y_pred)
    logger.info("[OK] Plot 1: Function approximation")
    
    # 2. Tangent Comparison
    viz.plot_tangent_comparison(
        x, y_true, tangent_results['x0'],
        tangent_results['true_tangent'],
        tangent_results['ai_tangent']
    )
    logger.info("[OK] Plot 2: Tangent comparison")
    
    # 3. Training Loss
    viz.plot_training_loss(training_results['history'])
    logger.info("[OK] Plot 3: Training loss")
    
    # 4. Derivative Comparison
    viz.plot_derivative_comparison(
        x, deriv_results['true_derivative'],
        deriv_results['autograd_derivative'],
        deriv_results['finite_diff_derivative']
    )
    logger.info("[OK] Plot 4: Derivative comparison")
    
    # 5. Error Analysis
    viz.plot_error_analysis(
        x, y_true, y_pred,
        deriv_results['true_derivative'],
        deriv_results['autograd_derivative']
    )
    logger.info("[OK] Plot 5: Error analysis")
    
    # 6. Noise Comparison
    model.eval()
    model_noisy.eval()
    with torch.no_grad():
        x_t = torch.FloatTensor(x.reshape(-1, 1)).to(device)
        y_pred_clean = model(x_t).cpu().numpy().flatten()
        y_pred_noisy = model_noisy(x_t).cpu().numpy().flatten()
    
    viz.plot_noise_comparison(
        x, data['y_clean'], data['y_noisy'],
        y_pred_clean, y_pred_noisy
    )
    logger.info("[OK] Plot 6: Noise comparison")
    
    # 7. Metrics Dashboard
    viz.plot_metrics_dashboard(metrics_results['metrics'])
    logger.info("[OK] Plot 7: Metrics dashboard")
    
    # 8. PINN Comparison
    model_pinn.eval()
    with torch.no_grad():
        y_pred_pinn = model_pinn(x_t).cpu().numpy().flatten()
    
    viz.plot_pinn_comparison(
        x, y_true, y_pred, y_pred_pinn,
        training_results['history'],
        training_results['history_pinn']
    )
    logger.info("[OK] Plot 8: PINN comparison")
    
    logger.info(f"All {8} visualizations saved to {config['figures_dir']}/")


# =============================================================
# Phase 7: Circle Experiment
# =============================================================

def phase_circle(config: dict, device: torch.device) -> dict:
    """Run the implicit circle experiment."""
    logger.info("=" * 60)
    logger.info("PHASE 7: IMPLICIT CIRCLE EXPERIMENT")
    logger.info("=" * 60)
    
    circle_results = run_circle_experiment(
        radius=config['circle_radius'],
        x0=config['circle_x0'],
        n_samples=config['circle_n_samples'],
        epochs=config['circle_epochs'],
        device=device
    )
    
    # Visualize circle
    viz = Visualizer(output_dir=config['figures_dir'])
    
    viz.plot_circle(
        radius=config['circle_radius'],
        x0=config['circle_x0'],
        y0=circle_results['y0'],
        tangent_data=circle_results['true_tangent'],
        ai_tangent_data=circle_results['ai_tangent'],
        nn_x=circle_results['x_plot'],
        nn_y=circle_results['y_pred']
    )
    logger.info("[OK] Plot: Circle visualization")
    
    viz.plot_implicit_tangent(
        radius=config['circle_radius'],
        x0=config['circle_x0'],
        y0=circle_results['y0'],
        true_tangent=circle_results['true_tangent'],
        ai_tangent=circle_results['ai_tangent']
    )
    logger.info("[OK] Plot: Implicit tangent")
    
    # Print circle metrics
    logger.info("\nCircle Experiment Results:")
    for k, v in circle_results['metrics'].items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.6e}")
    
    return circle_results


# =============================================================
# Main Execution
# =============================================================

def main() -> None:
    """Execute the complete AI function approximation pipeline."""
    
    print("\n" + "=" * 70)
    print("  AI-Based Neural Network System for Function Approximation")
    print("  Derivative Computation & Tangent Line Estimation")
    print("=" * 70)
    print("  Student: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)")
    print("  University: Sheikh Ayaz University")
    print("  Department: Computer Science Department")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    # Setup
    create_directories()
    set_seed(CONFIG['seed'])
    device = get_device()
    
    # Phase 1: Dataset Generation
    data = phase_dataset_generation(CONFIG)
    
    # Phase 2: Model Training
    training_results = phase_model_training(CONFIG, data, device)
    
    # Phase 3: Derivatives
    deriv_results = phase_derivatives(
        CONFIG, data, training_results['model'], device
    )
    
    # Phase 4: Tangent Lines
    tangent_results = phase_tangent_lines(
        CONFIG, data, training_results['model'], device
    )
    
    # Phase 5: Metrics
    metrics_results = phase_metrics(
        data, training_results['model'], device,
        tangent_results, deriv_results
    )
    
    # Phase 6: Visualization
    phase_visualization(
        CONFIG, data,
        training_results['model'],
        training_results['model_noisy'],
        training_results['model_pinn'],
        device, training_results,
        deriv_results, tangent_results, metrics_results
    )
    
    # Phase 7: Circle Experiment
    circle_results = phase_circle(CONFIG, device)
    
    # Final Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("  EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f} seconds")
    print(f"  Function MSE: {metrics_results['metrics']['function_mse']:.6e}")
    print(f"  Function R²:  {metrics_results['metrics']['function_r_squared']:.6f}")
    print(f"  Slope Error:  {metrics_results['metrics']['slope_error']:.6e}")
    print(f"  Circle MSE:   {circle_results['metrics']['function_mse']:.6e}")
    print(f"\n  Figures saved to: {CONFIG['figures_dir']}/")
    print(f"  Models saved to:  {CONFIG['models_dir']}/")
    print(f"  Metrics saved to: outputs/metrics.json")
    print(f"  Log saved to:     outputs/run.log")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
