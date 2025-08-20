
# Inventory Manager CLI Tool

A powerful command-line tool for processing inventory data and generating intelligent ordering recommendations.

## Features

- **Automatic File Detection**: Automatically finds inventory Excel files in your directory
- **Dynamic Column Recognition**: Detects stock and PO balance columns regardless of date format
- **Smart Ordering Algorithm**: Calculates optimal order quantities based on demand patterns, lead times, and MOQ constraints
- **Comprehensive Reports**: Generates detailed Excel output with color-coded recommendations and summary reports
- **Easy to Use**: Simple command-line interface with sensible defaults

## Requirements

- pandas>=1.3.0
- numpy>=1.20.0
- openpyxl>=3.0.0


## Installation

### Method 1: Direct Installation (Recommended)
```bash
pip install inventory-manager-cli
```

### Method 2: From Source
```bash
git clone <repository-url>
cd inventory-manager
pip install -e .
```

### Method 3: Standalone Script
Simply download `inventory_cli.py` and run it directly:
```bash
python inventory_cli.py
```

## Quick Start

1. **Place your inventory Excel file in a directory**
2. **Run the tool**:
   ```bash
   inventory-manager
   ```
3. **View results**: The tool will generate timestamped output files with recommendations

## Usage

### Basic Usage
```bash
# Process inventory file in current directory
inventory-manager

# Use short alias
inv-mgr
```

### Advanced Usage
```bash
# Process specific file
inventory-manager -i inventory_data.xlsx

# Specify input and output directories
inventory-manager -i /path/to/data/ -o /path/to/results/

# Enable verbose output
inventory-manager --verbose

# Save summary report to file
inventory-manager --save-report summary.txt

# Skip summary report
inventory-manager --no-report
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file or directory (default: current directory) |
| `-o, --output` | Output directory (default: current directory) |
| `-v, --verbose` | Enable verbose output |
| `--version` | Show version information |
| `--no-report` | Skip generating summary report |
| `--save-report FILE` | Save summary report to specified file |

## File Requirements

### Input File Format
- **Format**: Excel (.xlsx) files
- **File naming**: Files matching patterns like `inventory*.xlsx`, `*inventory*.xlsx`, `stock*.xlsx`, etc.
- **Required columns**:
  - Supplier
  - sanwa item
  - sanwa item code
  - RM (Raw Material)
  - Color
  - Revised Leadtime
  - Monthly demand columns (NOV'24, DEC'24, etc.)
  - Stock on hand column (auto-detected)
  - PO balance column (auto-detected)

### Output Files
- **Main output**: `{original_filename}_ordering_recommendations_{timestamp}.xlsx`
- **Summary report**: Console output or saved text file
- **Color coding**:
  - 🟡 Pink: First order column
  - 🔵 Blue: Other order columns  
  - 🟢 Green: Inventory level columns

## Algorithm Overview

The tool implements an intelligent inventory ordering system that:

1. **Analyzes demand patterns** from historical data
2. **Calculates safety stock** based on demand variability
3. **Considers lead times** for order scheduling
4. **Respects MOQ constraints** from supplier requirements
5. **Optimizes order timing** to minimize stockouts and carrying costs

### Key Features:
- **Dynamic reorder points** based on future demand
- **Safety stock calculation** using statistical methods
- **MOQ compliance** with automatic quantity adjustments
- **Lead time consideration** for order delivery scheduling

## Examples

### Example 1: Basic Processing
```bash
$ inventory-manager
[14:30:15] ℹ️ Inventory Management System initialized
[14:30:15] ℹ️ Found input file: inventory_data.xlsx
[14:30:15] ℹ️ Output will be saved as: inventory_data_ordering_recommendations_20241119_143015.xlsx
[14:30:16] ℹ️ Successfully read 150 rows from inventory_data.xlsx
[14:30:16] ℹ️ Using stock column: Sanwa stock on hand 27/11/2024
[14:30:16] ℹ️ Using PO balance column: Sanwa system PO bal as at 
[14:30:16] ℹ️ Calculating ordering recommendations...
[14:30:17] ℹ️ Saving results to: inventory_data_ordering_recommendations_20241119_143015.xlsx
[14:30:17] ✅ SUCCESS: Ordering recommendations saved to: inventory_data_ordering_recommendations_20241119_143015.xlsx
```

### Example 2: Verbose Mode with Custom Paths
```bash
$ inventory-manager -i /