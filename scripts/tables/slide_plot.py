import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# choose a color palette that is contrastive
pal = sns.color_palette(palette='Paired')

def create_beir_plot(df, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df, palette=pal)

    # set the y-axis from 50-60
    plt.ylim(50, 60)
    
    plt.title('BEIR (OOD)', fontsize=18, pad=20)
    plt.xlabel('Dataset', fontsize=14, labelpad=10)
    plt.ylabel('nDCG@10', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Model', title_fontsize='16', fontsize='13', loc='upper left')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=10, padding=3)
    
    plt.tight_layout()
    plt.savefig(f'plots/beir_comparison_plot_{figsize[0]}_{figsize[1]}.png', dpi=300, bbox_inches='tight')
    print(f"BEIR plot saved as 'plots/beir_comparison_plot_{figsize[0]}_{figsize[1]}.png'")

def create_ir_plot(df, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df, palette=pal)
    
    plt.title('Instruction Following', fontsize=18, pad=20)
    plt.xlabel('Dataset', fontsize=14, labelpad=10)
    plt.ylabel('Score', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    # rotate
    plt.xticks(rotation=30)
    plt.yticks(fontsize=12)
    plt.legend(title='Model', title_fontsize='12', fontsize='11', loc='upper left')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=10, padding=1)
    
    plt.tight_layout()
    plt.savefig(f'plots/ir_comparison_plot_{figsize[0]}_{figsize[1]}.png', dpi=300, bbox_inches='tight')
    print(f"IR plot saved as 'plots/ir_comparison_plot_{figsize[0]}_{figsize[1]}.png'")

# Prepare the data
data = {
    'Model': ['RepLLaMA', 'RepLLaMA', 'RepLLaMA', 'RepLLaMA', 'RepLLaMA',
              'Promptriever', 'Promptriever', 'Promptriever', 'Promptriever', 'Promptriever'],
    'Metric': ['No Prompt', 'w/Prompt', 'FollowIR MAP', 'FollowIR p-MRR', 'InstructIR Robust@10',
               'No Prompt', 'w/Prompt', 'FollowIR MAP', 'FollowIR p-MRR', 'InstructIR Robust@10'],
    'Value': [54.9, 54.8, 23.0, -3.1, 50.2,
              55.0, 56.4, 26.1, 11.2, 63.1]
}

df = pd.DataFrame(data)

# Set the style for the plots
sns.set_style("white")

# Create BEIR plot
beir_df = df[df['Metric'].isin(['No Prompt', 'w/Prompt'])]
create_beir_plot(beir_df)

# Create IR plot
ir_df = df[df['Metric'].isin(['FollowIR MAP', 'FollowIR p-MRR', 'InstructIR Robust@10'])]
# ir_df = ir_df[~((ir_df['Model'] == 'RepLLaMA') & (ir_df['Metric'] == 'Robust@10'))]
create_ir_plot(ir_df)

# now make the plots again but with a thinner plot size
create_beir_plot(beir_df, figsize=(4, 6))
create_ir_plot(ir_df, figsize=(4, 6))