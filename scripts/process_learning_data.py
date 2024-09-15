import pandas as pd

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'Outputs.csv')
df = pd.read_csv(file_path, header=None)
df.columns = ['x', 'y', 'name']
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df_filtered = df[df['x'] >= 1900]
max_x_value = df_filtered['x'].max()
max_x_df = df_filtered[df_filtered['x'] == max_x_value]
ground_truth_df = df_filtered[df_filtered['name'].str.contains('ground truth')]
other_data_df = max_x_df[~max_x_df['name'].str.contains('ground truth')]
ground_truth_df['y'] = pd.to_numeric(ground_truth_df['y'], errors='coerce')
other_data_df['y'] = pd.to_numeric(other_data_df['y'], errors='coerce')
differences = []
for action in other_data_df['name'].unique():
    ground_truth_y = ground_truth_df[ground_truth_df['name'] == f'{action} ground truth']['y'].values[0]
    action_rows = other_data_df[other_data_df['name'] == action]
    for index, row in action_rows.iterrows():
        y_diff = abs(row['y'] - ground_truth_y)
        differences.append({'x': row['x'], 'y_diff': y_diff, 'name': row['name']})
differences_df = pd.DataFrame(differences)
summary_stats = differences_df.groupby('name')['y_diff'].agg(['mean', 'std']).reset_index()
excel_file_path = '/Users/jamesmaskill/Documents/Filtered_and_Compared_Outputs_with_Stats.xlsx'
overall_stats = differences_df['y_diff'].agg(['mean', 'std']).reset_index()
overall_stats.columns = ['Metric', 'Value']

with pd.ExcelWriter(excel_file_path) as writer:
    other_data_df.to_excel(writer, sheet_name='Data', index=False)
    ground_truth_df.to_excel(writer, sheet_name='Ground Truth', index=False)
    differences_df.to_excel(writer, sheet_name='Y Differences', index=False)
    summary_stats.to_excel(writer, sheet_name='Summary Stats', index=False)
    overall_stats.to_excel(writer, sheet_name='Overall Stats', index=False)

print(f"Filtered and compared data with summary statistics saved to {excel_file_path}")
