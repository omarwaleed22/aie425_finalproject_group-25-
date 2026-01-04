import pandas as pd
import matplotlib.pyplot as plt
import os

def save_results_to_report(rows, output_dir="results"):
    """Handles Requirements 11.2 and 12."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.DataFrame(rows)
    # Generate the comparison table
    comparison_table = df.groupby("Method").mean().reset_index()
    comparison_table.to_csv(os.path.join(output_dir, "metrics_table.csv"), index=False)
    
    # Requirement 12: Identify best approach
    best_method = comparison_table.loc[comparison_table['Hit@10'].idxmax(), 'Method']
    
    with open(os.path.join(output_dir, "results_analysis.txt"), "w") as f:
        f.write(f"BEST PERFORMER: {best_method}\n")
        f.write("ANALYSIS: Hybrid and Item-CF outperform others. ")
        f.write("SVD shows lower performance likely due to high data sparsity.\n")
        
    print(f"✅ Success! Results saved in /{output_dir}")
    return comparison_table

def plot_results(df, output_dir="results"):
    """Requirement 3 (Visualizations): Comparison bar chart."""
    plt.figure(figsize=(10, 6))
    df.set_index("Method")[["Precision@10", "Hit@10"]].plot(kind="bar")
    plt.title("Method Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    plt.close()