import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def parse_date(date_str):
    """Convert date string to datetime object"""
    try:
        # Handle different date formats
        if 'Q' in str(date_str):  # Handle quarters like "Q1 2023"
            quarter, year = date_str.split()
            month = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}[quarter]
            return datetime(int(year), month, 1)
        else:
            # Handle "Mon YYYY" format
            return datetime.strptime(date_str, '%b %Y')
    except:
        # Fallback for any other formats
        try:
            return datetime.strptime(date_str, '%B %Y')
        except:
            print(f"Could not parse date: {date_str}")
            return None

def map_owner_to_category(owner):
    """Map owner names to the specified categories"""
    owner_lower = owner.lower()
    
    # Define mappings
    if 'microsoft' in owner_lower or owner == 'Open AI / Microsoft':
        return 'microsoft'
    elif 'google' in owner_lower or 'deepmind' in owner_lower:
        return 'google'
    elif 'meta' in owner_lower or 'facebook' in owner_lower:
        return 'meta'
    elif 'anthropic' in owner_lower:
        return 'anthropic'
    elif 'mistral' in owner_lower:
        return 'mistral'
    elif 'openai' in owner_lower:
        return 'openai'
    elif 'xai' in owner_lower or owner == 'Twitter':  # xAI is Elon's AI company (formerly Twitter)
        return 'xai'
    else:
        return 'other'

def is_major_llm(name, params):
    """Determine if an LLM is major/important enough to include"""
    name_lower = name.lower()
    
    # Major model families and important milestones
    major_models = [
        'bert', 'gpt', 'claude', 'llama', 'palm', 'gemini', 'chatgpt',
        't5', 'wu dao', 'megatron', 'bloom', 'opt', 'chinchilla',
        'lamda', 'pathways', 'flamingo', 'gopher', 'jurassic',
        'codex', 'davinci', 'instruct', 'turbo', 'bard', 'vicuna',
        'alpaca', 'mixtral', 'yi', 'qwen', 'ernie', 'baichuan'
    ]
    
    # Check if name contains any major model keywords
    is_major_by_name = any(major in name_lower for major in major_models)
    
    # Include if it's a major model by name OR if it's very large (>100B params)
    is_large = params >= 100
    
    return is_major_by_name or is_large

def create_llm_scatterplot(csv_file_path):
    """Create a scatterplot of LLM model sizes over time"""
    
    # Read the CSV file
    # Skip the first few rows that contain metadata
    try:
        df = pd.read_csv(csv_file_path, skiprows=2)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Trying without skipping rows...")
        df = pd.read_csv(csv_file_path)
    
    # Print original columns to debug
    print("Original columns:", df.columns.tolist())
    
    # Clean column names - handle different possible column names
    if len(df.columns) >= 4:
        df.columns = ['name', 'owner', 'parameters_billions', 'date', 'notes', 'link'] if len(df.columns) >= 6 else ['name', 'owner', 'parameters_billions', 'date']
    else:
        print("Warning: Unexpected number of columns:", len(df.columns))
        print("Columns:", df.columns.tolist())
        
    print(f"Data shape: {df.shape}")
    print("First few rows:")
    print(df.head())
    
    # Clean and convert parameters column
    def clean_parameters(param_str):
        """Clean and convert parameter values to float"""
        if pd.isna(param_str):
            return None
        
        # Convert to string and clean
        param_str = str(param_str).strip()
        
        # Handle asterisks and other indicators
        if '*' in param_str:
            param_str = param_str.replace('*', '')
        
        # Remove commas from numbers (like "1,500" -> "1500")
        param_str = param_str.replace(',', '')
        
        # Handle ranges (take the average or first value)
        if '-' in param_str and param_str.replace('-', '').replace('.', '').isdigit():
            # Handle ranges like "1.3-1.7"
            parts = param_str.split('-')
            try:
                return float(parts[0])  # Take first value
            except:
                return None
        
        # Handle other text indicators
        param_str = param_str.replace('~', '').replace('>', '').replace('<', '')
        param_str = param_str.replace('billion', '').replace('B', '').strip()
        
        try:
            return float(param_str)
        except:
            return None
    
    # Apply cleaning function
    df['parameters_billions_clean'] = df['parameters_billions'].apply(clean_parameters)
    
    # Print some debugging info
    print(f"\nCleaning results:")
    print(f"Original values sample: {df['parameters_billions'].head(10).tolist()}")
    print(f"Cleaned values sample: {df['parameters_billions_clean'].head(10).tolist()}")
    print(f"Non-null cleaned values: {df['parameters_billions_clean'].notna().sum()}")
    
    # Filter out rows with missing essential data
    df_clean = df.dropna(subset=['name', 'owner', 'parameters_billions_clean', 'date'])
    df_clean = df_clean[df_clean['parameters_billions_clean'] > 0]  # Remove zero or negative values
    
    print(f"Rows after cleaning: {len(df_clean)}")
    
    # Filter to only major/important LLMs
    df_clean['is_major'] = df_clean.apply(lambda row: is_major_llm(row['name'], row['parameters_billions_clean']), axis=1)
    df_clean = df_clean[df_clean['is_major']]
    
    print(f"Rows after filtering to major LLMs: {len(df_clean)}")
    
    # Parse dates
    df_clean['parsed_date'] = df_clean['date'].apply(parse_date)
    df_clean = df_clean.dropna(subset=['parsed_date'])
    
    # Map owners to categories
    df_clean['category'] = df_clean['owner'].apply(map_owner_to_category)
    
    # Create the plot with larger figure size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 24,           # Base font size
        'axes.titlesize': 30,      # Title font size
        'axes.labelsize': 26,      # Axis label font size
        'xtick.labelsize': 34,     # X-axis tick font size
        'ytick.labelsize': 34,     # Y-axis tick font size
        'legend.fontsize': 22,     # Legend font size
        'figure.titlesize': 34     # Figure title font size
    })
    
    # Define more vivid colors for each category
    colors = {
        'microsoft': '#FF6B35',    # Vibrant orange-red
        'google': '#4285F4',       # Google blue (already vivid)
        'meta': '#FF3366',         # Bright pink-red
        'anthropic': '#FF8C00',    # Dark orange
        'mistral': '#9D4EDD',      # Bright purple
        'openai': '#00D9FF',       # Cyan
        'xai': '#1A1A1A',          # Dark for contrast
        'other': '#32CD32'         # Lime green
    }
    
    # Calculate larger sizes for circles based on parameter count
    # Use sqrt scaling to make differences visible while not overwhelming
    min_params = df_clean['parameters_billions_clean'].min()
    max_params = df_clean['parameters_billions_clean'].max()
    
    def calculate_circle_size(params):
        # Scale circle sizes between 100 and 1000 pixels (much larger than before)
        # Use square root scaling for better visual distribution
        normalized = (np.sqrt(params) - np.sqrt(min_params)) / (np.sqrt(max_params) - np.sqrt(min_params))
        return 100 + normalized * 900
    
    # Create scatter plot for each category with consistent legend circle size
    legend_circle_size = 200  # Fixed size for legend circles
    
    for category in colors.keys():
        data = df_clean[df_clean['category'] == category]
        if len(data) > 0:
            sizes = data['parameters_billions_clean'].apply(calculate_circle_size)
            # Use actual sizes for scatter plot
            ax.scatter(data['parsed_date'], data['parameters_billions_clean'], 
                      c=colors[category], alpha=0.8, s=sizes, 
                      edgecolors='white', linewidth=1.5)
            # Add invisible scatter with fixed size for legend
            ax.scatter([], [], c=colors[category], alpha=0.8, s=legend_circle_size,
                      edgecolors='white', linewidth=1.5, label=category.upper())
    
    # Add labels only for the most famous models
    famous_models = ['BERT', 'GPT-2', 'GPT-3', 'GPT-4', 'T5', 'ChatGPT', 'Claude', 'LLaMA', 'PaLM', 'Gemini', 'Grok', 'Wu Dao', 'BLOOM', 'Chinchilla', 'LaMDA']
    
    for _, row in df_clean.iterrows():
        # Check if this model should be labeled
        should_label = any(famous in row['name'] for famous in famous_models)
        
        if should_label:
            ax.annotate(row['name'], 
                       (row['parsed_date'], row['parameters_billions_clean']),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=11, alpha=0.9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8, edgecolor='gray'))
    
    # Customize the plot
    ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('Model Size (Billion Parameters)', fontsize=18, fontweight='bold')
    ax.set_title('The Evolution of Major Large Language Models: Size Over Time', 
                fontsize=22, fontweight='bold', pad=25)
    
    # Set y-axis to log scale for better visualization
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(fontsize=22)
    
    # Add main legend with more spacing
    main_legend = ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', 
                       frameon=True, fancybox=True, fontsize=14,
                       title='Company', title_fontsize=16,
                       borderpad=1.5, handletextpad=1.2, columnspacing=2)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Print some statistics
    print(f"Total major models plotted: {len(df_clean)}")
    print(f"Date range: {df_clean['parsed_date'].min().strftime('%b %Y')} to {df_clean['parsed_date'].max().strftime('%b %Y')}")
    print(f"Parameter range: {df_clean['parameters_billions_clean'].min():.2f} to {df_clean['parameters_billions_clean'].max():.0f} billion")
    
    category_counts = df_clean['category'].value_counts()
    print("\nMajor models per category:")
    for cat, count in category_counts.items():
        print(f"  {cat.upper()}: {count}")
    
    plt.show()
    
    return fig, df_clean


# Usage example:
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_path = "data/llm_data.csv"
    
    try:
        fig, data = create_llm_scatterplot(csv_path)
        
        # Save as PDF
        pdf_filename = "llm_evolution_scatterplot.pdf"
        fig.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Plot saved as PDF: {pdf_filename}")
        
        print("Plot created successfully!")
    except FileNotFoundError:
        print(f"CSV file not found. Please make sure the file path is correct.")
    except Exception as e:
        print(f"Error creating plot: {e}")