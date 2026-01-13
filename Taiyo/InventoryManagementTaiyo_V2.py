#!/usr/bin/env python3
"""
Resin Inventory Management System v2.0
A tool for processing resin BOM data and generating ordering recommendations.

Key Features:
- Two-level aggregation: First at Finished Good (FG) level, then at Resin PartNo level
- Lead time consideration for ordering recommendations
- Safety stock calculation based on demand variability and lead time
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

__version__ = "2.0.0"
__author__ = "Resin Inventory Management System"


class ResinInventoryManager:
    """Main resin inventory management class with FG-level aggregation and lead time support"""

    # Define the unique key columns for Finished Goods
    FG_KEY_COLUMNS = ['Model', 'MP or Non-MP', 'FG_PN ', 'JIG']

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.log("Resin Inventory Management System v2.0 initialized")

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

    def clean_string(self, value):
        """Clean and standardize string values"""
        if pd.isna(value):
            return None
        return str(value).strip()

    def identify_demand_columns(self, df):
        """Identify demand forecast columns"""
        demand_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if 'demand' in col_lower and ('week' in col_lower or 'forecast' in col_lower or 'forcast' in col_lower):
                demand_cols.append(col)
        return sorted(demand_cols, key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'))

    def read_resin_bom(self, file_path):
        """Read the Resin BOM sheet from Excel file"""
        try:
            df = pd.read_excel(file_path, sheet_name='Resin_BOM')
            self.log(f"Successfully read {len(df)} rows from Resin_BOM sheet")
            return df
        except Exception as e:
            self.log(f"Failed to read Resin_BOM sheet: {e}", "ERROR")
            raise

    def create_fg_key(self, row):
        """Create a unique key for each Finished Good"""
        parts = []
        for col in self.FG_KEY_COLUMNS:
            val = row.get(col, '')
            if pd.notna(val):
                parts.append(str(val).strip())
            else:
                parts.append('')
        return '|'.join(parts)

    def aggregate_at_fg_level(self, df, demand_cols):
        """
        First level aggregation: Aggregate data at the Finished Good level
        Each FG is unique by: Model, MP or Non-MP, FG_PN, JIG
        """
        self.log("Step 1: Aggregating at Finished Good (FG) level...")

        # Create unique FG key
        df['FG_Key'] = df.apply(self.create_fg_key, axis=1)

        # Get weight column
        weight_col = 'Colour weight\n (g)'
        leadtime_col = 'Lead time (weeks)'

        # Convert weight to kg and calculate demand in kg for each row
        df['Weight_kg'] = df[weight_col].apply(lambda x: self.safe_numeric_convert(x) / 1000)

        # Calculate demand in kg for each week
        for col in demand_cols:
            df[f'{col}_kg'] = df.apply(
                lambda row: self.safe_numeric_convert(row[col])/row['CAV'] * row['Weight_kg'],
                axis=1
            )

        # Convert inventory columns
        df['Current_Inventory_kg'] = df['Current Resin inventory (g)'].apply(
            lambda x: self.safe_numeric_convert(x) / 1000
        )
        df['Open_Order_kg'] = df['Resin open order (g)'].apply(
            lambda x: self.safe_numeric_convert(x) / 1000
        )
        df['FG_Inventory_units'] = df['Current finish good inventory (units)'].apply(
            lambda x: self.safe_numeric_convert(x)
        )
        df['Lead_Time_Weeks'] = df[leadtime_col].apply(
            lambda x: self.safe_numeric_convert(x, default=4)
        )

        # Store original data with FG keys for later use
        df['Resin_PartNo_Clean'] = df['Resin PartNo'].apply(self.clean_string)

        self.log(f"   Found {df['FG_Key'].nunique()} unique Finished Goods")

        return df

    def aggregate_at_resin_level(self, df, demand_cols):
        """
        Second level aggregation: Aggregate from FG level to Resin PartNo level
        Sum up demands from all FGs that use the same resin
        """
        self.log("Step 2: Aggregating at Resin PartNo level...")

        # Filter out rows with no resin
        df = df[df['Resin_PartNo_Clean'].notna()].copy()

        # Build aggregation dictionary
        agg_dict = {
            'Customer': lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v)))),
            'Model': lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v)))),
            'FG_PN ': lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v)))),
            'FG_Key': lambda x: len(set(x)),  # Count of unique FGs using this resin
            'Weight_kg': 'mean',
            'Current_Inventory_kg': 'sum',
            'Open_Order_kg': 'sum',
            'FG_Inventory_units': 'sum',
            'Lead_Time_Weeks': 'max',  # Use maximum lead time for safety
        }

        # Add demand columns to aggregation (sum across all FGs)
        for col in demand_cols:
            agg_dict[f'{col}_kg'] = 'sum'

        # Group by Resin PartNo
        aggregated = df.groupby('Resin_PartNo_Clean').agg(agg_dict).reset_index()
        aggregated.rename(columns={
            'Resin_PartNo_Clean': 'Resin PartNo',
            'FG_Key': 'Num_FGs_Using_Resin'
        }, inplace=True)

        self.log(f"   Aggregated to {len(aggregated)} unique Resin PartNo values")

        return aggregated

    def calculate_safety_stock_with_leadtime(self, demands, lead_time_weeks, service_level=0.95):
        """
        Calculate safety stock based on demand variability AND lead time

        Safety Stock = Z * σ * √(Lead Time)
        where:
        - Z = Z-score for desired service level (1.65 for 95%)
        - σ = standard deviation of demand
        - Lead Time = in same units as demand period (weeks)
        """
        clean_demands = [self.safe_numeric_convert(d) for d in demands if self.safe_numeric_convert(d) > 0]
        if len(clean_demands) == 0:
            return 0

        # Z-score for service level
        z_score = 1.65 if service_level == 0.95 else 1.28

        # Calculate demand variability
        demand_std = np.std(clean_demands) if len(clean_demands) > 1 else np.mean(clean_demands) * 0.2

        # Safety stock increases with square root of lead time
        # Lead time factor: √(lead_time_weeks)
        lead_time_factor = np.sqrt(max(1, lead_time_weeks))

        safety_stock = z_score * demand_std * lead_time_factor

        return max(0, safety_stock)

    def calculate_demand_during_leadtime(self, row, demand_cols, lead_time_weeks):
        """
        Calculate expected demand during lead time period

        If lead time is 5 weeks and we have 4 weeks of forecast data,
        we extrapolate based on average weekly demand
        """
        weekly_demands = []
        for col in demand_cols:
            kg_col = f'{col}_kg'
            if kg_col in row.index:
                weekly_demands.append(self.safe_numeric_convert(row[kg_col]))

        if not weekly_demands:
            return 0

        avg_weekly_demand = np.mean(weekly_demands)
        demand_during_leadtime = avg_weekly_demand * lead_time_weeks

        return demand_during_leadtime

    def calculate_ordering_recommendation(self, row, demand_cols):
        """
        Calculate ordering recommendation considering lead time

        The reorder point considers:
        1. Demand during lead time period
        2. Safety stock (which increases with longer lead time)
        3. Current inventory and open orders
        """
        # Get lead time
        lead_time_weeks = self.safe_numeric_convert(row.get('Lead_Time_Weeks', 4))

        # Calculate weekly demands
        weekly_demands = []
        for col in demand_cols:
            kg_col = f'{col}_kg'
            if kg_col in row.index:
                weekly_demands.append(self.safe_numeric_convert(row[kg_col]))

        # Total demand over forecast period (typically 4 weeks = 1 month)
        monthly_demand = sum(weekly_demands)

        # Average weekly demand
        avg_weekly_demand = monthly_demand / len(weekly_demands) if weekly_demands else 0

        # Demand during lead time
        demand_during_leadtime = avg_weekly_demand * lead_time_weeks

        # Get current inventory and open orders
        current_inventory = self.safe_numeric_convert(row.get('Current_Inventory_kg', 0))
        open_orders = self.safe_numeric_convert(row.get('Open_Order_kg', 0))

        # Available stock = current inventory + incoming orders
        available_stock = current_inventory + open_orders

        # Calculate safety stock (considers lead time)
        safety_stock = self.calculate_safety_stock_with_leadtime(
            weekly_demands, lead_time_weeks
        )

        # Reorder Point = Demand during lead time + Safety Stock
        reorder_point = demand_during_leadtime + safety_stock

        # Order Quantity = max(0, Reorder Point - Available Stock)
        # Also ensure we cover at least the monthly demand
        min_order_qty = max(0, monthly_demand - available_stock)
        order_qty_from_rop = max(0, reorder_point - available_stock)

        # Take the larger of the two to ensure adequate coverage
        order_qty = max(min_order_qty, order_qty_from_rop)

        # Projected ending inventory after order arrives and monthly demand consumed
        projected_ending = available_stock + order_qty - monthly_demand

        # Determine urgency based on lead time and stock coverage
        stock_coverage_weeks = available_stock / avg_weekly_demand if avg_weekly_demand > 0 else float('inf')

        if stock_coverage_weeks < lead_time_weeks:
            urgency = "URGENT - Order Immediately"
        elif stock_coverage_weeks < lead_time_weeks * 1.5:
            urgency = "HIGH - Order Soon"
        elif order_qty > 0:
            urgency = "NORMAL - Plan Order"
        else:
            urgency = "LOW - Sufficient Stock"

        return {
            'lead_time_weeks': lead_time_weeks,
            'avg_weekly_demand_kg': avg_weekly_demand,
            'monthly_demand_kg': monthly_demand,
            'demand_during_leadtime_kg': demand_during_leadtime,
            'current_inventory_kg': current_inventory,
            'open_orders_kg': open_orders,
            'available_stock_kg': available_stock,
            'safety_stock_kg': safety_stock,
            'reorder_point_kg': reorder_point,
            'recommended_order_kg': order_qty,
            'projected_ending_inventory_kg': projected_ending,
            'stock_coverage_weeks': stock_coverage_weeks,
            'urgency': urgency
        }

    def process_inventory(self, input_file_path, output_file_path=None):
        """
        Main function to process inventory and generate ordering recommendations

        Process Flow:
        1. Read BOM data
        2. Aggregate at FG level (unique by Model, MP or Non-MP, FG_PN, JIG)
        3. Aggregate at Resin PartNo level
        4. Calculate ordering recommendations with lead time consideration
        """
        self.log(f"Processing resin inventory from: {input_file_path}")

        # Read the BOM data
        df = self.read_resin_bom(input_file_path)

        # Identify demand columns
        demand_cols = self.identify_demand_columns(df)
        self.log(f"Found {len(demand_cols)} demand forecast columns: {demand_cols}")

        # Step 1: Aggregate at FG level
        df_fg = self.aggregate_at_fg_level(df, demand_cols)

        # Step 2: Aggregate at Resin level
        aggregated = self.aggregate_at_resin_level(df_fg, demand_cols)

        # Step 3: Calculate ordering recommendations
        self.log("Step 3: Calculating ordering recommendations with lead time...")
        recommendations = []

        for idx, row in aggregated.iterrows():
            calc = self.calculate_ordering_recommendation(row, demand_cols)

            rec = {
                'Resin_PartNo': row['Resin PartNo'],
                'Num_FGs_Using': row['Num_FGs_Using_Resin'],
                'Customers': row['Customer'],
                'Models': row['Model'],
                'Lead_Time_Weeks': calc['lead_time_weeks'],
                'Avg_Weight_per_Unit_kg': round(row['Weight_kg'], 6),
                'Week1_Demand_kg': round(self.safe_numeric_convert(row.get(f'{demand_cols[0]}_kg', 0)), 3) if len(
                    demand_cols) > 0 else 0,
                'Week2_Demand_kg': round(self.safe_numeric_convert(row.get(f'{demand_cols[1]}_kg', 0)), 3) if len(
                    demand_cols) > 1 else 0,
                'Week3_Demand_kg': round(self.safe_numeric_convert(row.get(f'{demand_cols[2]}_kg', 0)), 3) if len(
                    demand_cols) > 2 else 0,
                'Week4_Demand_kg': round(self.safe_numeric_convert(row.get(f'{demand_cols[3]}_kg', 0)), 3) if len(
                    demand_cols) > 3 else 0,
                'Avg_Weekly_Demand_kg': round(calc['avg_weekly_demand_kg'], 3),
                'Monthly_Total_Demand_kg': round(calc['monthly_demand_kg'], 3),
                'Demand_During_LeadTime_kg': round(calc['demand_during_leadtime_kg'], 3),
                'Current_Inventory_kg': round(calc['current_inventory_kg'], 3),
                'Open_Orders_kg': round(calc['open_orders_kg'], 3),
                'Available_Stock_kg': round(calc['available_stock_kg'], 3),
                'Safety_Stock_kg': round(calc['safety_stock_kg'], 3),
                'Reorder_Point_kg': round(calc['reorder_point_kg'], 3),
                'Recommended_Order_kg': round(calc['recommended_order_kg'], 3),
                'Projected_Ending_Inventory_kg': round(calc['projected_ending_inventory_kg'], 3),
                'Stock_Coverage_Weeks': round(calc['stock_coverage_weeks'], 1) if calc['stock_coverage_weeks'] != float(
                    'inf') else 'Infinite',
                'Order_Urgency': calc['urgency']
            }
            recommendations.append(rec)

        # Create output DataFrame
        output_df = pd.DataFrame(recommendations)

        # Sort by urgency and then by order quantity
        urgency_order = {
            'URGENT - Order Immediately': 0,
            'HIGH - Order Soon': 1,
            'NORMAL - Plan Order': 2,
            'LOW - Sufficient Stock': 3
        }
        output_df['_urgency_sort'] = output_df['Order_Urgency'].map(urgency_order)
        output_df = output_df.sort_values(
            ['_urgency_sort', 'Recommended_Order_kg'],
            ascending=[True, False]
        ).drop('_urgency_sort', axis=1)

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
        print("\n" + "=" * 90)
        print("RESIN INVENTORY ORDERING SUMMARY REPORT (with Lead Time Analysis)")
        print("=" * 90)

        print(f"\n📊 OVERVIEW:")
        print(f"   Total unique Resin PartNo: {len(output_df)}")

        # Count by urgency
        urgency_counts = output_df['Order_Urgency'].value_counts()
        print(f"\n   Order Urgency Breakdown:")
        for urgency, count in urgency_counts.items():
            print(f"      {urgency}: {count}")

        total_demand = output_df['Monthly_Total_Demand_kg'].sum()
        total_order = output_df['Recommended_Order_kg'].sum()
        print(f"\n   Total monthly demand: {total_demand:.2f} kg")
        print(f"   Total recommended order: {total_order:.2f} kg")

        # Lead time distribution
        print("\n   Lead Time Distribution:")
        lt_dist = output_df['Lead_Time_Weeks'].value_counts().sort_index()
        for lt, count in lt_dist.items():
            print(f"      {lt} weeks: {count} resins")

        print("\n" + "-" * 90)
        print("🔴 URGENT ORDERS (Stock < Lead Time Coverage):")
        print("-" * 90)
        urgent = output_df[output_df['Order_Urgency'] == 'URGENT - Order Immediately']
        if len(urgent) > 0:
            for _, row in urgent.head(10).iterrows():
                resin_name = row['Resin_PartNo'][:45] + "..." if len(str(row['Resin_PartNo'])) > 45 else row[
                    'Resin_PartNo']
                print(f"   {resin_name}")
                print(
                    f"      Lead Time: {row['Lead_Time_Weeks']} wks | Order: {row['Recommended_Order_kg']:.1f} kg | Coverage: {row['Stock_Coverage_Weeks']} wks")
        else:
            print("   No urgent orders!")

        print("\n" + "-" * 90)
        print("🟡 HIGH PRIORITY ORDERS:")
        print("-" * 90)
        high = output_df[output_df['Order_Urgency'] == 'HIGH - Order Soon']
        if len(high) > 0:
            for _, row in high.head(10).iterrows():
                resin_name = row['Resin_PartNo'][:45] + "..." if len(str(row['Resin_PartNo'])) > 45 else row[
                    'Resin_PartNo']
                print(f"   {resin_name}")
                print(
                    f"      Lead Time: {row['Lead_Time_Weeks']} wks | Order: {row['Recommended_Order_kg']:.1f} kg | Coverage: {row['Stock_Coverage_Weeks']} wks")
        else:
            print("   No high priority orders!")

        print("\n" + "-" * 90)
        print("📋 TOP 10 BY ORDER QUANTITY:")
        print("-" * 90)
        top_orders = output_df[output_df['Recommended_Order_kg'] > 0].nlargest(10, 'Recommended_Order_kg')
        for _, row in top_orders.iterrows():
            resin_name = row['Resin_PartNo'][:45] + "..." if len(str(row['Resin_PartNo'])) > 45 else row['Resin_PartNo']
            print(f"   {resin_name}")
            print(
                f"      Order: {row['Recommended_Order_kg']:.1f} kg | Demand/Mo: {row['Monthly_Total_Demand_kg']:.1f} kg | LT: {row['Lead_Time_Weeks']} wks")

        print("\n" + "=" * 90)
        return output_df


def main():
    """Main entry point"""
    import sys

    manager = ResinInventoryManager(verbose=True)

    # Default input file path
    input_file = "Resin_BOM_list_test_new.xlsx"
    output_file = "resin_ordering_recommendations_v2.csv"

    # Check if custom input file is provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Process the inventory
    result_df = manager.process_inventory(
        input_file_path=input_file,
        output_file_path=output_file
    )

    # Generate summary report
    manager.generate_summary_report(result_df)

    return result_df


if __name__ == "__main__":
    main()