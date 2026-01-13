#!/usr/bin/env python3
"""
Resin Inventory Management System
A tool for processing resin BOM data and generating ordering recommendations.
Aggregates demand forecasts by Resin PartNo across all customers and products.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

__version__ = "1.0.0"
__author__ = "Resin Inventory Management System"


class ResinInventoryManager:
    """Main resin inventory management class"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.log("Resin Inventory Management System initialized")

    def log(self, message, level="INFO"):
        """Simple logging function"""
        if self.verbose or level in ["ERROR", "SUCCESS"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = "✅" if level == "SUCCESS" else "❌" if level == "ERROR" else "ℹ️"
            print(f"[{timestamp}] {prefix} {message}")

    def safe_numeric_convert(self, value, default=0):
        """Safely convert a value to numeric, handling NaN and invalid values"""
        if pd.isna(value):
            return default
        try:
            result = float(value)
            return max(0, result)
        except (ValueError, TypeError):
            return default

    def clean_resin_name(self, name):
        """Clean and standardize resin part number names"""
        if pd.isna(name):
            return None
        return str(name).strip()

    def calculate_safety_stock(self, demands, service_level=0.95):
        """Calculate safety stock based on demand variability"""
        clean_demands = [self.safe_numeric_convert(d) for d in demands if self.safe_numeric_convert(d) > 0]
        if len(clean_demands) == 0:
            return 0

        demand_std = np.std(clean_demands) if len(clean_demands) > 1 else np.mean(clean_demands) * 0.2
        # Z-score for 95% service level is approximately 1.65
        z_score = 1.65 if service_level == 0.95 else 1.28
        safety_stock = z_score * demand_std

        return max(0, safety_stock)

    def read_resin_bom(self, file_path):
        """Read the Resin BOM sheet from Excel file"""
        try:
            df = pd.read_excel(file_path, sheet_name='Resin_BOM')
            self.log(f"Successfully read {len(df)} rows from Resin_BOM sheet")
            return df
        except Exception as e:
            self.log(f"Failed to read Resin_BOM sheet: {e}", "ERROR")
            raise

    def identify_demand_columns(self, df):
        """Identify demand forecast columns"""
        demand_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if 'demand' in col_lower and ('week' in col_lower or 'forecast' in col_lower or 'forcast' in col_lower):
                demand_cols.append(col)
        return demand_cols

    def aggregate_by_resin(self, df):
        """
        Aggregate demand and inventory data by unique Resin PartNo
        Each Resin PartNo gets total demand from all customers/products
        """
        resin_col = 'Resin PartNo'
        weight_col = 'Colour weight\n (g)'

        # Find demand columns
        demand_cols = self.identify_demand_columns(df)
        self.log(f"Found {len(demand_cols)} demand forecast columns: {demand_cols}")

        # Clean resin names
        df['Resin_Clean'] = df[resin_col].apply(self.clean_resin_name)
        df = df[df['Resin_Clean'].notna()]

        # Convert weight to kg (from grams)
        df['Weight_kg'] = df[weight_col].apply(lambda x: self.safe_numeric_convert(x) / 1000)

        # Convert demand from units to kg for each week - original
        for col in demand_cols:
            df[f'{col}_kg'] = df.apply(
                lambda row: self.safe_numeric_convert(row[col]) * row['Weight_kg'],
                axis=1
            )

        #todo: here need to consider CAV and finish goods in stock!!!
        # Convert demand from units to kg for each week - consider CAV
        # Todo: not correct, this is for every week, should NOT - current FG for every week!!!!
        # for col in demand_cols:
        #     df[f'{col}_kg'] = df.apply(
        #         lambda row: (self.safe_numeric_convert(row[col])-self.safe_numeric_convert(row['Current finish good inventory (units)']))
        #                     /self.safe_numeric_convert(row['CAV']) * row['Weight_kg'],
        #         axis=1
        #     )

        # Also convert current inventory columns
        df['Current_Inventory_kg'] = df['Current Resin inventory (g)'].apply(
            lambda x: self.safe_numeric_convert(x) / 1000
        )
        df['Open_Order_kg'] = df['Resin open order (g)'].apply(
            lambda x: self.safe_numeric_convert(x) / 1000
        )

        # Aggregate by Resin PartNo
        agg_dict = {
            'Customer': lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v)))),
            'Model': lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v)))),
            'Weight_kg': 'mean',  # Average weight per unit
            'Current_Inventory_kg': 'sum',
            'Open_Order_kg': 'sum',
        }

        # Add demand columns to aggregation
        for col in demand_cols:
            agg_dict[f'{col}_kg'] = 'sum'

        # Group by Resin PartNo
        aggregated = df.groupby('Resin_Clean').agg(agg_dict).reset_index()
        aggregated.rename(columns={'Resin_Clean': 'Resin PartNo'}, inplace=True)

        self.log(f"Aggregated data for {len(aggregated)} unique Resin PartNo values")
        return aggregated, demand_cols

    def calculate_monthly_demand(self, row, demand_cols):
        """Calculate monthly demand from weekly forecasts (4 weeks = 1 month)"""
        total_demand = 0
        for col in demand_cols:
            kg_col = f'{col}_kg'
            if kg_col in row.index:
                total_demand += self.safe_numeric_convert(row[kg_col])
        return total_demand

    def calculate_ordering_recommendation(self, row, demand_cols, leadtime_weeks=4, safety_factor=1.2):
        """
        Calculate ordering recommendation for a single resin

        Parameters:
        - leadtime_weeks: Lead time for ordering (default 4 weeks = 1 month)
        - safety_factor: Multiplier for safety stock (default 1.2 = 20% buffer)
        """
        # Calculate total monthly demand (sum of all weeks)
        monthly_demand = self.calculate_monthly_demand(row, demand_cols)

        # Get current inventory and open orders
        current_inventory = self.safe_numeric_convert(row.get('Current_Inventory_kg', 0))
        open_orders = self.safe_numeric_convert(row.get('Open_Order_kg', 0))

        # Available stock = current inventory + incoming orders
        available_stock = current_inventory + open_orders

        # Calculate safety stock (based on demand variability)
        demand_values = [self.safe_numeric_convert(row.get(f'{col}_kg', 0)) for col in demand_cols]
        safety_stock = self.calculate_safety_stock(demand_values) * safety_factor

        # Reorder point = monthly demand + safety stock
        reorder_point = monthly_demand + safety_stock

        # Order quantity = max(0, reorder_point - available_stock)
        order_qty = max(0, reorder_point - available_stock)

        # Calculate projected ending inventory
        projected_ending = available_stock - monthly_demand + order_qty

        return {
            'monthly_demand_kg': monthly_demand,
            'current_inventory_kg': current_inventory,
            'open_orders_kg': open_orders,
            'available_stock_kg': available_stock,
            'safety_stock_kg': safety_stock,
            'reorder_point_kg': reorder_point,
            'recommended_order_kg': order_qty,
            'projected_ending_inventory_kg': projected_ending
        }

    def process_inventory(self, input_file_path, output_file_path=None,
                          leadtime_weeks=4, safety_factor=1.2):
        """
        Main function to process inventory and generate ordering recommendations

        Parameters:
        - input_file_path: Path to the Excel file with Resin BOM data
        - output_file_path: Path for output CSV (optional, auto-generated if not provided)
        - leadtime_weeks: Lead time for ordering
        - safety_factor: Safety stock multiplier
        """
        self.log(f"Processing resin inventory from: {input_file_path}")

        # Read the BOM data
        df = self.read_resin_bom(input_file_path)

        # Aggregate by Resin PartNo
        aggregated, demand_cols = self.aggregate_by_resin(df)

        # Calculate ordering recommendations for each resin
        self.log("Calculating ordering recommendations...")
        recommendations = []

        for idx, row in aggregated.iterrows():
            calc = self.calculate_ordering_recommendation(
                row, demand_cols, leadtime_weeks, safety_factor
            )

            rec = {
                'Resin PartNo': row['Resin PartNo'],
                'Customers': row['Customer'],
                'Models': row['Model'],
                'Avg_Weight_per_Unit_kg': round(row['Weight_kg'], 6),
                'Week1_Demand_kg': round(self.safe_numeric_convert(row.get(f'{demand_cols[0]}_kg', 0)), 3) if len(
                    demand_cols) > 0 else 0,
                'Week2_Demand_kg': round(self.safe_numeric_convert(row.get(f'{demand_cols[1]}_kg', 0)), 3) if len(
                    demand_cols) > 1 else 0,
                'Week3_Demand_kg': round(self.safe_numeric_convert(row.get(f'{demand_cols[2]}_kg', 0)), 3) if len(
                    demand_cols) > 2 else 0,
                'Week4_Demand_kg': round(self.safe_numeric_convert(row.get(f'{demand_cols[3]}_kg', 0)), 3) if len(
                    demand_cols) > 3 else 0,
                'Monthly_Total_Demand_kg': round(calc['monthly_demand_kg'], 3),
                'Current_Inventory_kg': round(calc['current_inventory_kg'], 3),
                'Open_Orders_kg': round(calc['open_orders_kg'], 3),
                'Available_Stock_kg': round(calc['available_stock_kg'], 3),
                'Safety_Stock_kg': round(calc['safety_stock_kg'], 3),
                'Reorder_Point_kg': round(calc['reorder_point_kg'], 3),
                'Recommended_Order_kg': round(calc['recommended_order_kg'], 3),
                'Projected_Ending_Inventory_kg': round(calc['projected_ending_inventory_kg'], 3),
                'Order_Status': 'ORDER NEEDED' if calc['recommended_order_kg'] > 0 else 'SUFFICIENT STOCK'
            }
            recommendations.append(rec)

        # Create output DataFrame
        output_df = pd.DataFrame(recommendations)

        # Sort by order recommendation (highest first)
        output_df = output_df.sort_values('Recommended_Order_kg', ascending=False)

        # Generate output file path if not provided
        if output_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_path = f"resin_ordering_recommendations_{timestamp}.csv"

        # Save to CSV
        output_df.to_csv(output_file_path, index=False)
        self.log(f"Recommendations saved to: {output_file_path}", "SUCCESS")

        return output_df

    def generate_summary_report(self, output_df):
        """Generate a summary report of the ordering recommendations"""
        print("\n" + "=" * 80)
        print("RESIN INVENTORY ORDERING SUMMARY REPORT")
        print("=" * 80)

        print(f"\n📊 OVERVIEW:")
        print(f"   Total unique Resin PartNo: {len(output_df)}")
        print(f"   Resins requiring orders: {len(output_df[output_df['Recommended_Order_kg'] > 0])}")
        print(f"   Resins with sufficient stock: {len(output_df[output_df['Recommended_Order_kg'] == 0])}")

        total_demand = output_df['Monthly_Total_Demand_kg'].sum()
        total_order = output_df['Recommended_Order_kg'].sum()
        print(f"\n   Total monthly demand: {total_demand:.2f} kg")
        print(f"   Total recommended order: {total_order:.2f} kg")

        print("\n" + "-" * 80)
        print("🔴 TOP 10 RESINS BY ORDER QUANTITY:")
        print("-" * 80)
        top_orders = output_df[output_df['Recommended_Order_kg'] > 0].head(10)
        if len(top_orders) > 0:
            for _, row in top_orders.iterrows():
                resin_name = row['Resin PartNo'][:50] + "..." if len(str(row['Resin PartNo'])) > 50 else row[
                    'Resin PartNo']
                print(f"   {resin_name}")
                print(
                    f"      Order: {row['Recommended_Order_kg']:.2f} kg | Demand: {row['Monthly_Total_Demand_kg']:.2f} kg")
        else:
            print("   No orders needed - all resins have sufficient stock!")

        print("\n" + "-" * 80)
        print("🟢 RESINS WITH SUFFICIENT STOCK:")
        print("-" * 80)
        sufficient = output_df[output_df['Recommended_Order_kg'] == 0]
        if len(sufficient) > 0:
            for _, row in sufficient.head(5).iterrows():
                resin_name = row['Resin PartNo'][:50] + "..." if len(str(row['Resin PartNo'])) > 50 else row[
                    'Resin PartNo']
                print(f"   {resin_name}")
                print(
                    f"      Available: {row['Available_Stock_kg']:.2f} kg | Demand: {row['Monthly_Total_Demand_kg']:.2f} kg")
            if len(sufficient) > 5:
                print(f"   ... and {len(sufficient) - 5} more")

        print("\n" + "=" * 80)
        return output_df


def main():
    """Main entry point"""
    import sys

    manager = ResinInventoryManager(verbose=True)

    # Default input file path
    input_file = "Resin_BOM_list_test.xlsx"
    output_file = "resin_ordering_recommendations.csv"

    # Check if custom input file is provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Process the inventory
    result_df = manager.process_inventory(
        input_file_path=input_file,
        output_file_path=output_file,
        leadtime_weeks=4,
        safety_factor=1.2
    )

    # Generate summary report
    manager.generate_summary_report(result_df)

    return result_df


if __name__ == "__main__":
    main()