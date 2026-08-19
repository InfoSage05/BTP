import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set aesthetic style
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'axes.grid': True})

results_file = r'D:\BTP\results\MASTER_results_table.csv'
data_file = r'D:\BTP\data\chf_long_clean.csv'
output_dir = r'D:\BTP\results\figures_paper'

os.makedirs(output_dir, exist_ok=True)

# 1. Load Results
df_res = pd.read_csv(results_file)
df_res = df_res.dropna(subset=['C_r2_seed42'])

# Sort by C_r2_seed42 for better visualization
df_res_sorted = df_res.sort_values(by='C_r2_seed42', ascending=False).head(15) # Top 15 models

# 2. Bar Plot for R2 Scores
plt.figure(figsize=(12, 8))
models = df_res_sorted['model (target)'].values
y_pos = np.arange(len(models))

width = 0.25
plt.barh(y_pos + width, df_res_sorted['A_r2'], width, label='Split A', color='skyblue')
plt.barh(y_pos, df_res_sorted['B_r2'], width, label='Split B', color='orange')
plt.barh(y_pos - width, df_res_sorted['C_r2_seed42'], width, label='Split C', color='green')

plt.yticks(y_pos, models)
plt.xlabel('R² Score', fontsize=14)
plt.title('R² Score Comparison Across Data Splits (Top 15 Models)', fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_r2_comparison.png'))
plt.close()

# 3. MAPE Comparison Bar Chart
plt.figure(figsize=(10, 6))
df_mape = df_res_sorted.sort_values(by='C_mape', ascending=False)
plt.barh(df_mape['model (target)'], df_mape['C_mape'], color='coral')
plt.title('MAPE Comparison on Split C (Lower is Better)', fontsize=16, fontweight='bold')
plt.xlabel('Mean Absolute Percentage Error (%)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_mape_comparison.png'))
plt.close()

# 4. Load Data for Heatmap
df_data = pd.read_csv(data_file)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_data['X'], df_data['G'], c=df_data['CHF'], cmap='magma', alpha=0.7, s=20)
plt.colorbar(scatter, label='Critical Heat Flux (CHF)')
plt.title('CHF Distribution based on Quality (X) and Mass Flux (G)', fontsize=16, fontweight='bold')
plt.xlabel('Thermodynamic Quality (X)', fontsize=14)
plt.ylabel('Mass Flux (G) [kg/m²s]', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chf_scatter_heatmap.png'))
plt.close()

# 5. Hexbin plot for density visualization
plt.figure(figsize=(10, 8))
plt.hexbin(df_data['X'], df_data['P'], C=df_data['CHF'], gridsize=30, cmap='inferno', reduce_C_function=np.mean)
plt.colorbar(label='Mean CHF')
plt.title('Average CHF across Pressure (P) and Quality (X)', fontsize=16, fontweight='bold')
plt.xlabel('Thermodynamic Quality (X)', fontsize=14)
plt.ylabel('Pressure (P) [kPa]', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chf_hexbin_heatmap.png'))
plt.close()

print("Figures generated successfully in:", output_dir)
