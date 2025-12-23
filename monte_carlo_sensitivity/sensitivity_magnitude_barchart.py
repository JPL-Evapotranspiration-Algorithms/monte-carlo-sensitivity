from typing import Union, List
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import pandas as pd

def sensitivity_magnitude_barchart(
        model_name: str,
        output_variable: Union[str, List[str]],
        df: pd.DataFrame,
        metric: str = "mean_normalized_change"):

    if isinstance(output_variable, str):
        output_variable = [output_variable]  # Convert to list for uniform handling

    num_vars = len(output_variable)
    fig, axes = plt.subplots(nrows=1, ncols=num_vars, figsize=(6 * num_vars, 6), constrained_layout=True)

    # Ensure axes is iterable even for a single subplot
    if num_vars == 1:
        axes = [axes]

    # Determine the global y-axis limits
    y_min, y_max = float('inf'), float('-inf')
    for var in output_variable:
        filtered_df = df[(df.output_variable == var) & (df.metric == metric)]
        y_min = min(y_min, (filtered_df.value * 100).min())
        y_max = max(y_max, (filtered_df.value * 100).max())

    for ax, var in zip(axes, output_variable):
        filtered_df = df[(df.output_variable == var) & (df.metric == metric)]
        filtered_df = filtered_df.sort_values(by="value", ascending=False)  # Sort bars in descending order

        sns.barplot(x=filtered_df.input_variable, y=filtered_df.value * 100, color='black', ax=ax)

        # Rotate x-axis labels 45 degrees
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        ax.set_xlabel("Input Variable")
        ax.set_ylabel("Average Percent Change in Output Perturbation")
        ax.set_title(f"{model_name} {var} Sensitivity Magnitude")
        ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5)  # Add light gray horizontal gridlines only

        # Add percent sign to y-axis tick labels
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}%'))

        # Set the global y-axis limits
        ax.set_ylim(y_min, y_max)

    # Save the figure
    plt.savefig(f"{model_name} Sensitivity Magnitude Multi-Panel.jpeg", format='jpeg', bbox_inches='tight')
    plt.savefig(f"{model_name} Sensitivity Magnitude Multi-Panel.svg", format='svg', bbox_inches='tight')

    plt.show()
