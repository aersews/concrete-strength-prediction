"""
Research Paper Visualization Generator
Concrete Compressive Strength Prediction
Generates publication-ready figures for academic paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Professional color palette
COLORS = {
    'CatBoost_Tuned': '#FF6B6B',  # Red - Best Model
    'CatBoost': '#FF8888',
    'LightGBM': '#4ECDC4',  # Teal
    'XGBoost': '#45B7D1',  # Blue
    'ExtraTrees': '#FFA07A',  # Light Salmon
    'RF_Tuned': '#98D8C8',  # Mint
    'MLP': '#F7DC6F',  # Yellow
    'RandomForest': '#BB8FCE',  # Lavender
    'GradientBoosting': '#85C1E2',  # Light Blue
    'DecisionTree': '#F8B88B',  # Peach
    'KNN': '#95E1D3',  # Aquamint
    'SVR': '#C7CEEA',  # Periwinkle
    'Ridge': '#B2DFDB',  # Pale Teal
    'Linear': '#D1C4E9',  # Lavender Blue
    'Lasso': '#FFCCBC',  # Peach
    'ElasticNet': '#FFE0B2',  # Light Peach
}

# Load data
baseline_df = pd.read_csv(os.path.join(OUT_DIR, "baseline_metrics.csv"))
leaderboard_df = pd.read_csv(os.path.join(OUT_DIR, "final_leaderboard.csv"))

print("📊 Generating Research Paper Visualizations...")
print(f"✓ Baseline Models: {len(baseline_df)}")
print(f"✓ Top Models: {len(leaderboard_df)}")

# ============================================================
# Figure 1: Model Performance Leaderboard (Main Results)
# ============================================================

def create_leaderboard_visual():
    """Figure 1: Leaderboard with ranking and key metrics"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    leaderboard_display = leaderboard_df.copy()
    
    # Sort by rank
    leaderboard_display = leaderboard_display.sort_values('Rank')
    
    # Create horizontal positions
    y_pos = np.arange(len(leaderboard_display))
    
    # Get colors
    colors = [COLORS.get(model, '#cccccc') for model in leaderboard_display['Model']]
    
    # Create bar chart for Test_R2 (main metric)
    bars = ax.barh(y_pos, leaderboard_display['Test_R2'], color=colors, edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(leaderboard_display['Model'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Test R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Leaderboard\nConcrete Compressive Strength Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0.85, 0.945)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, leaderboard_display['Test_R2'])):
        ax.text(value + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Add rank numbers
    for i, rank in enumerate(leaderboard_display['Rank']):
        ax.text(0.852, y_pos[i], f'#{int(rank)}', va='center', ha='right', 
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle', facecolor='darkgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '01_leaderboard.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_leaderboard.png")
    plt.close()


# ============================================================
# Figure 2: Multi-Metric Comparison (Top Models)
# ============================================================

def create_metrics_comparison():
    """Figure 2: Side-by-side comparison of all metrics"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    metrics = ['CV_R2', 'Test_R2', 'Adj_R2', 'MAE', 'RMSE']
    titles = ['Cross-Validation R² Score', 'Test R² Score', 'Adjusted R² Score',
              'Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)']
    
    leaderboard_display = leaderboard_df.sort_values('Test_R2', ascending=False)
    colors = [COLORS.get(model, '#cccccc') for model in leaderboard_display['Model']]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        if metric in ['MAE', 'RMSE']:
            # Lower is better
            bars = ax.bar(range(len(leaderboard_display)), leaderboard_display[metric], 
                         color=colors, edgecolor='black', linewidth=1.2)
            ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        else:
            # Higher is better
            bars = ax.bar(range(len(leaderboard_display)), leaderboard_display[metric], 
                         color=colors, edgecolor='black', linewidth=1.2)
            ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(range(len(leaderboard_display)))
        ax.set_xticklabels(leaderboard_display['Model'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Remove extra subplot
    fig.add_subplot(gs[1, 2]).axis('off')
    
    fig.suptitle('Comprehensive Model Performance Metrics Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(os.path.join(PLOT_DIR, '02_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_metrics_comparison.png")
    plt.close()


# ============================================================
# Figure 3: R² Score Progression (All Models)
# ============================================================

def create_r2_progression():
    """Figure 3: R² progression across all baseline models"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    baseline_sorted = baseline_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)
    x = np.arange(len(baseline_sorted))
    
    colors_baseline = [COLORS.get(model, '#cccccc') for model in baseline_sorted['Model']]
    
    # Plot lines and markers
    ax.plot(x, baseline_sorted['CV_R2'], marker='o', linewidth=2.5, markersize=8, 
            label='CV R²', color='#2E86AB', alpha=0.8)
    ax.plot(x, baseline_sorted['Test_R2'], marker='s', linewidth=2.5, markersize=8, 
            label='Test R²', color='#A23B72', alpha=0.8)
    ax.plot(x, baseline_sorted['Adj_R2'], marker='^', linewidth=2.5, markersize=8, 
            label='Adjusted R²', color='#F18F01', alpha=0.8)
    
    # Customize
    ax.set_xlabel('Models (Ranked by Test R²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('R² Score Evolution Across All Models\nBaseline Performance Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_sorted['Model'], rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0.5, 0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '03_r2_progression.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_r2_progression.png")
    plt.close()


# ============================================================
# Figure 4: Error Metrics Comparison (MAE vs RMSE)
# ============================================================

def create_error_metrics():
    """Figure 4: Error metrics comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    baseline_sorted = baseline_df.sort_values('Test_R2', ascending=False)
    colors_baseline = [COLORS.get(model, '#cccccc') for model in baseline_sorted['Model']]
    x_pos = np.arange(len(baseline_sorted))
    
    # MAE comparison
    bars1 = axes[0].bar(x_pos, baseline_sorted['MAE'], color=colors_baseline, 
                        edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Mean Absolute Error (MPa)', fontsize=11, fontweight='bold')
    axes[0].set_title('MAE: Lower is Better', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(baseline_sorted['Model'], rotation=45, ha='right', fontsize=8)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_axisbelow(True)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    # RMSE comparison
    bars2 = axes[1].bar(x_pos, baseline_sorted['RMSE'], color=colors_baseline, 
                        edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Root Mean Square Error (MPa)', fontsize=11, fontweight='bold')
    axes[1].set_title('RMSE: Lower is Better', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(baseline_sorted['Model'], rotation=45, ha='right', fontsize=8)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_axisbelow(True)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    fig.suptitle('Error Metrics Analysis: All Baseline Models', 
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_error_metrics.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_error_metrics.png")
    plt.close()


# ============================================================
# Figure 5: Model Category Performance Tree
# ============================================================

def create_model_category_performance():
    """Figure 5: Models grouped by category"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define categories
    categories = {
        'Tree-Based Ensemble': ['CatBoost_Tuned', 'CatBoost', 'LightGBM', 'XGBoost', 'ExtraTrees', 'RF_Tuned', 'RandomForest', 'GradientBoosting', 'DecisionTree'],
        'Linear Models': ['Linear', 'Ridge', 'Lasso', 'ElasticNet'],
        'Instance-Based': ['KNN'],
        'Neural/SVM': ['MLP', 'SVR']
    }
    
    category_colors = {
        'Tree-Based Ensemble': '#FF6B6B',
        'Linear Models': '#4ECDC4',
        'Instance-Based': '#F7DC6F',
        'Neural/SVM': '#95E1D3'
    }
    
    # Prepare data
    models_in_data = baseline_df['Model'].tolist()
    
    category_data = []
    x_labels = []
    colors_plot = []
    
    for category, models in categories.items():
        for model in models:
            if model in models_in_data:
                row = baseline_df[baseline_df['Model'] == model].iloc[0]
                category_data.append((model, row['Test_R2'], category))
                x_labels.append(model)
                colors_plot.append(category_colors[category])
    
    # Sort by Test_R2
    category_data.sort(key=lambda x: x[1], reverse=True)
    x_labels = [item[0] for item in category_data]
    y_values = [item[1] for item in category_data]
    colors_final = [category_colors[item[2]] for item in category_data]
    
    x_pos = np.arange(len(x_labels))
    bars = ax.bar(x_pos, y_values, color=colors_final, edgecolor='black', linewidth=1.2)
    
    # Customize
    ax.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance by Category\nTree-Based Ensemble Dominance in Regression Tasks', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat) 
                      for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '05_model_category_performance.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_model_category_performance.png")
    plt.close()


# ============================================================
# Figure 6: Top 7 Models - Detailed Metrics Heatmap
# ============================================================

def create_metrics_heatmap():
    """Figure 6: Heatmap of normalized metrics for top models"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    top_models = leaderboard_df.sort_values('Rank').head(7)
    
    # Select metrics to display
    metrics_to_show = ['CV_R2', 'Test_R2', 'Adj_R2', 'MAE', 'RMSE']
    data_for_heatmap = top_models[metrics_to_show].copy()
    
    # Normalize each column to 0-1 scale
    data_normalized = data_for_heatmap.copy()
    for col in data_normalized.columns:
        if col in ['MAE', 'RMSE']:
            # For error metrics, invert so that lower is better (appears darker)
            min_val = data_normalized[col].min()
            max_val = data_normalized[col].max()
            data_normalized[col] = 1 - (data_normalized[col] - min_val) / (max_val - min_val)
        else:
            # For R2 scores, keep as is
            min_val = data_normalized[col].min()
            max_val = data_normalized[col].max()
            data_normalized[col] = (data_normalized[col] - min_val) / (max_val - min_val)
    
    # Create heatmap
    sns.heatmap(data_normalized, annot=data_for_heatmap.values, fmt='.4f', 
                cmap='RdYlGn', cbar_kws={'label': 'Normalized Performance'}, ax=ax,
                xticklabels=metrics_to_show, yticklabels=top_models['Model'],
                linewidths=0.5, linecolor='black', cbar=True, vmin=0, vmax=1)
    
    ax.set_title('Performance Metrics Heatmap - Top 7 Models\n(Normalized: Green=Better, Red=Worse)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '06_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_metrics_heatmap.png")
    plt.close()


# ============================================================
# Figure 7: Test R² vs MAE Trade-off Analysis
# ============================================================

def create_performance_tradeoff():
    """Figure 7: Scatter plot showing accuracy vs error trade-off"""
    fig, ax = plt.subplots(figsize=(13, 8))
    
    baseline_sorted = baseline_df.sort_values('Test_R2', ascending=False)
    
    # Create scatter plot
    scatter = ax.scatter(baseline_sorted['MAE'], baseline_sorted['Test_R2'], 
                        s=300, alpha=0.7, edgecolor='black', linewidth=1.5,
                        c=range(len(baseline_sorted)), cmap='viridis')
    
    # Add model labels
    for idx, row in baseline_sorted.iterrows():
        ax.annotate(row['Model'], (row['MAE'], row['Test_R2']), 
                   fontsize=9, fontweight='bold', ha='center', va='center')
    
    # Highlight best models
    best_models = leaderboard_df.head(3)
    best_in_baseline = baseline_sorted.head(3)
    
    ax.scatter(best_in_baseline['MAE'], best_in_baseline['Test_R2'], 
              s=500, facecolors='none', edgecolors='red', linewidth=2.5,
              label='Top 3 Models', zorder=5)
    
    ax.set_xlabel('Mean Absolute Error (MAE) - MPa [Lower is Better]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test R² Score [Higher is Better]', fontsize=12, fontweight='bold')
    ax.set_title('Performance Trade-off: Accuracy vs Error\nOptimal Region: High R², Low MAE', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc='lower right')
    
    # Add quadrant annotations
    mid_mae = baseline_sorted['MAE'].median()
    mid_r2 = baseline_sorted['Test_R2'].median()
    ax.axhline(mid_r2, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(mid_mae, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '07_performance_tradeoff.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 07_performance_tradeoff.png")
    plt.close()


# ============================================================
# Figure 8: Tuning Impact - Before vs After
# ============================================================

def create_tuning_impact():
    """Figure 8: Compare baseline vs tuned versions of same models"""
    fig, ax = plt.subplots(figsize=(13, 7))
    
    # Get models that have tuned versions
    tuned_models = leaderboard_df[leaderboard_df['Model'].str.contains('Tuned', case=False)]['Model'].tolist()
    
    tuning_data = []
    for tuned_model in tuned_models:
        base_model = tuned_model.replace('_Tuned', '')
        if base_model in baseline_df['Model'].values:
            tuned_r2 = leaderboard_df[leaderboard_df['Model'] == tuned_model]['Test_R2'].values[0]
            base_r2 = baseline_df[baseline_df['Model'] == base_model]['Test_R2'].values[0]
            improvement = (tuned_r2 - base_r2) / base_r2 * 100
            tuning_data.append({
                'Base Model': base_model,
                'Baseline R²': base_r2,
                'Tuned R²': tuned_r2,
                'Improvement %': improvement
            })
    
    if len(tuning_data) > 0:
        tuning_df = pd.DataFrame(tuning_data)
        
        x = np.arange(len(tuning_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, tuning_df['Baseline R²'], width, label='Baseline', 
                      color='#FF8888', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, tuning_df['Tuned R²'], width, label='Tuned', 
                      color='#FF6B6B', edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Hyperparameter Tuning Impact\nComparison of Baseline vs Optimized Models', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(tuning_df['Base Model'], fontsize=11, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add improvement percentages
        for i, (idx, row) in enumerate(tuning_df.iterrows()):
            ax.text(i, max(row['Baseline R²'], row['Tuned R²']) + 0.003, 
                   f"+{row['Improvement %']:.2f}%", ha='center', fontsize=10, 
                   fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '08_tuning_impact.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: 08_tuning_impact.png")
        plt.close()


# ============================================================
# Figure 9: Distribution Analysis - Top Models
# ============================================================

def create_cv_vs_test_comparison():
    """Figure 9: CV R² vs Test R² - Generalization analysis"""
    fig, ax = plt.subplots(figsize=(13, 8))
    
    baseline_sorted = baseline_df.sort_values('Test_R2', ascending=False)
    x_pos = np.arange(len(baseline_sorted))
    width = 0.35
    
    colors_cv = ['#2E86AB' if baseline_sorted.iloc[i]['CV_R2'] < baseline_sorted.iloc[i]['Test_R2'] 
                 else '#A23B72' for i in range(len(baseline_sorted))]
    
    bars1 = ax.bar(x_pos - width/2, baseline_sorted['CV_R2'], width, 
                  label='Cross-Validation R²', color='#2E86AB', edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, baseline_sorted['Test_R2'], width, 
                  label='Test R²', color='#A23B72', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Draw diagonal line indicating perfect parity
    ax.plot([-1, len(baseline_sorted)], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Parity')
    
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Generalization Analysis: Cross-Validation vs Test Performance\nAll Baseline Models', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(baseline_sorted['Model'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(0.55, 0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '09_cv_vs_test.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 09_cv_vs_test.png")
    plt.close()


# ============================================================
# Figure 10: Summary Statistics Table
# ============================================================

def create_summary_table():
    """Figure 10: Statistical summary of all models"""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create comprehensive summary
    summary_stats = baseline_df[['Model', 'CV_R2', 'Test_R2', 'Adj_R2', 'MAE', 'RMSE']].copy()
    summary_stats = summary_stats.sort_values('Test_R2', ascending=False)
    
    # Add rank
    summary_stats.insert(0, 'Rank', range(1, len(summary_stats) + 1))
    
    # Format numbers
    for col in ['CV_R2', 'Test_R2', 'Adj_R2', 'MAE', 'RMSE']:
        summary_stats[col] = summary_stats[col].apply(lambda x: f'{x:.4f}')
    
    # Create table
    table = ax.table(cellText=summary_stats.values, colLabels=summary_stats.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_stats.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_stats) + 1):
        for j in range(len(summary_stats.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
            
            # Highlight top 3 models
            if i <= 3:
                table[(i, j)].set_facecolor('#D4EDDA')
    
    plt.title('Comprehensive Model Performance Table\nAll 14 Baseline Models', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '10_summary_table.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 10_summary_table.png")
    plt.close()


# ============================================================
# Generate All Visualizations
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Research Paper Visualization Generator")
    print("="*60 + "\n")
    
    create_leaderboard_visual()
    create_metrics_comparison()
    create_r2_progression()
    create_error_metrics()
    create_model_category_performance()
    create_metrics_heatmap()
    create_performance_tradeoff()
    create_tuning_impact()
    create_cv_vs_test_comparison()
    create_summary_table()
    
    print("\n" + "="*60)
    print("✅ All visualizations generated successfully!")
    print("="*60)
    print(f"\n📁 Output Directory: {PLOT_DIR}")
    print("\n📊 Generated Figures:")
    print("   1. 01_leaderboard.png - Model ranking by Test R² Score")
    print("   2. 02_metrics_comparison.png - All metrics comparison (5 subplots)")
    print("   3. 03_r2_progression.png - R² evolution across models")
    print("   4. 04_error_metrics.png - MAE and RMSE comparison")
    print("   5. 05_model_category_performance.png - Performance by model category")
    print("   6. 06_metrics_heatmap.png - Normalized metrics heatmap (Top 7)")
    print("   7. 07_performance_tradeoff.png - Accuracy vs Error scatter plot")
    print("   8. 08_tuning_impact.png - Baseline vs Tuned comparison")
    print("   9. 09_cv_vs_test.png - Generalization analysis")
    print("   10. 10_summary_table.png - Statistical summary table")
    print("\n" + "="*60 + "\n")
