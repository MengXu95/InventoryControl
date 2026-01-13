#!/usr/bin/env python3
"""
Resin Inventory Management System v3.0
A tool for processing resin BOM data and generating ordering recommendations.

Key Features:
- Proper FG inventory deduction: NET demand = Total Demand - FG Inventory on hand
- Only NET demand (production requirement) is converted to kg for resin calculation
- Two-level aggregation: First at Finished Good (FG) level, then at Resin PartNo level
- Lead time consideration for ordering recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

__version__ = "3.0.0"
__author__ = "Resin Inventory Management System"


class ResinInventoryManager:
    """Main resin inventory management class with proper FG inventory handling"""

    # Define the unique key columns for Finished Goods
    FG_KEY_COLUMNS = ['Model', 'MP or Non-MP', 'FG_PN ', 'JIG', 'Resin PartNo']

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.log("Resin Inventory Management System v3.0 initialized")

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

    def calculate_net_demand_at_fg_level(self, df, demand_cols):
        """
        Step 1: Calculate NET demand at FG level

        Logic:
        - Total Demand (units) = Sum of all weekly forecasts
        - FG Inventory on Hand (units) = Current finished goods in stock
        - NET Demand (units) = max(0, Total Demand - FG Inventory)
        - This NET demand is what actually needs to be PRODUCED, thus requiring resin
        """
        self.log("Step 1: Calculating NET demand at Finished Good (FG) level...")

        # Create unique FG key
        df['FG_Key'] = df.apply(self.create_fg_key, axis=1)

        # print("df['FG_Key']:", len(df['FG_Key']))

        # Get column names
        weight_col = 'Colour weight\n (g)'
        leadtime_col = 'Lead time (weeks)'
        fg_inventory_col = 'Current finish good inventory (units)'

        # Convert weight to kg
        df['Weight_kg'] = df[weight_col].apply(lambda x: self.safe_numeric_convert(x) / 1000)

        # Get FG inventory on hand (units)
        df['FG_Inventory_Units'] = df[fg_inventory_col].apply(self.safe_numeric_convert)

        # Calculate total demand in units (sum of all weeks)
        df['Total_Demand_Units'] = 0
        for col in demand_cols:
            df['Total_Demand_Units'] += df[col].apply(self.safe_numeric_convert)

        # Calculate weekly demands in units
        for i, col in enumerate(demand_cols):
            df[f'Week{i + 1}_Demand_Units'] = df[col].apply(self.safe_numeric_convert)

        # *** KEY CALCULATION ***
        # NET demand = Total Demand - FG Inventory on hand
        # This represents what actually needs to be PRODUCED
        df['Net_Demand_Units'] = df.apply(
            lambda row: max(0, row['Total_Demand_Units'] - row['FG_Inventory_Units']),
            axis=1
        )

        # Calculate the ratio of net demand to total demand (for proportional weekly breakdown)
        df['Net_Demand_Ratio'] = df.apply(
            lambda row: row['Net_Demand_Units'] / row['Total_Demand_Units']
            if row['Total_Demand_Units'] > 0 else 0,
            axis=1
        )

        # Convert NET demand to kg (this is the actual resin requirement)
        df['Net_Demand_kg'] = df['Net_Demand_Units']/df['CAV'] * df['Weight_kg']

        # Also calculate weekly net demands in kg (proportionally reduced)
        for i, col in enumerate(demand_cols):
            # Apply the net demand ratio to get proportional weekly net demand
            df[f'Week{i + 1}_Net_Demand_kg'] = df[f'Week{i + 1}_Demand_Units'] * df['Net_Demand_Ratio'] * df[
                'Weight_kg']

        # Get other inventory data
        df['Current_Resin_Inventory_kg'] = df['Current Resin inventory (g)'].apply(
            lambda x: self.safe_numeric_convert(x) / 1000
        )
        df['Resin_Open_Order_kg'] = df['Resin open order (g)'].apply(
            lambda x: self.safe_numeric_convert(x) / 1000
        )
        df['Lead_Time_Weeks'] = df[leadtime_col].apply(
            lambda x: self.safe_numeric_convert(x, default=4)
        )

        # Clean resin name
        df['Resin_PartNo_Clean'] = df['Resin PartNo'].apply(self.clean_string)

        # Log summary
        total_gross_demand = df['Total_Demand_Units'].sum()
        total_fg_inventory = df['FG_Inventory_Units'].sum()
        total_net_demand = df['Net_Demand_Units'].sum()

        self.log(f"   Total Gross Demand: {total_gross_demand:,.0f} units")
        self.log(f"   Total FG Inventory on Hand: {total_fg_inventory:,.0f} units")
        self.log(f"   Total NET Demand (needs production): {total_net_demand:,.0f} units")
        self.log(f"   Demand Reduction from FG Inventory: {(1 - total_net_demand / total_gross_demand) * 100:.1f}%")

        return df

    def aggregate_at_resin_level(self, df, demand_cols):
        """
        Step 2: Aggregate NET demand from FG level to Resin PartNo level
        Sum up NET demands (in kg) from all FGs that use the same resin
        """
        self.log("Step 2: Aggregating NET demand at Resin PartNo level...")

        # Filter out rows with no resin
        df = df[df['Resin_PartNo_Clean'].notna()].copy()

        # Build aggregation dictionary
        agg_dict = {
            'Customer': lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v)))),
            'Model': lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v)))),
            'FG_PN ': lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v)))),
            'FG_Key': lambda x: len(set(x)),  # Count of unique FGs using this resin
            'Weight_kg': 'mean',
            'Total_Demand_Units': 'sum',
            'FG_Inventory_Units': 'sum',
            'Net_Demand_Units': 'sum',
            'Net_Demand_kg': 'sum',
            'Current_Resin_Inventory_kg': 'sum',
            'Resin_Open_Order_kg': 'sum',
            'Lead_Time_Weeks': 'max',  # Use maximum lead time for safety
        }

        # Add weekly NET demand columns to aggregation
        for i in range(len(demand_cols)):
            agg_dict[f'Week{i + 1}_Net_Demand_kg'] = 'sum'

        # Group by Resin PartNo
        aggregated = df.groupby('Resin_PartNo_Clean').agg(agg_dict).reset_index()
        aggregated.rename(columns={
            'Resin_PartNo_Clean': 'Resin_PartNo',
            'FG_Key': 'Num_FGs_Using_Resin'
        }, inplace=True)

        self.log(f"   Aggregated to {len(aggregated)} unique Resin PartNo values")

        # Log summary
        total_net_demand_kg = aggregated['Net_Demand_kg'].sum()
        self.log(f"   Total NET Resin Requirement: {total_net_demand_kg:,.2f} kg")

        return aggregated

    def calculate_safety_stock_with_leadtime(self, weekly_demands, lead_time_weeks, service_level=0.95):
        """
        Calculate safety stock based on demand variability AND lead time

        Safety Stock = Z * σ * √(Lead Time)
        """
        clean_demands = [d for d in weekly_demands if d > 0]
        if len(clean_demands) == 0:
            return 0

        z_score = 1.65 if service_level == 0.95 else 1.28
        demand_std = np.std(clean_demands) if len(clean_demands) > 1 else np.mean(clean_demands) * 0.2
        lead_time_factor = np.sqrt(max(1, lead_time_weeks))
        safety_stock = z_score * demand_std * lead_time_factor

        return max(0, safety_stock)

    def calculate_ordering_recommendation(self, row, num_weeks):
        """
        Calculate ordering recommendation based on NET demand and lead time

        Key formulas:
        - Net Demand (kg) = Already calculated (gross demand - FG inventory, converted to kg)
        - Available Resin = Current Resin Inventory + Open Orders
        - Demand During Lead Time = Avg Weekly Net Demand × Lead Time
        - Safety Stock = Z × σ × √(Lead Time)
        - Reorder Point = Demand During Lead Time + Safety Stock
        - Order Qty = max(0, Reorder Point - Available Resin)
        """
        # Get lead time
        lead_time_weeks = self.safe_numeric_convert(row.get('Lead_Time_Weeks', 4))

        # Get weekly NET demands in kg
        weekly_net_demands = []
        for i in range(num_weeks):
            weekly_net_demands.append(self.safe_numeric_convert(row.get(f'Week{i + 1}_Net_Demand_kg', 0)))

        # Total NET demand for the month (already accounts for FG inventory)
        monthly_net_demand_kg = self.safe_numeric_convert(row.get('Net_Demand_kg', 0))

        # Average weekly NET demand
        avg_weekly_net_demand = monthly_net_demand_kg / num_weeks if num_weeks > 0 else 0

        # Demand during lead time (based on NET demand)
        demand_during_leadtime = avg_weekly_net_demand * lead_time_weeks

        # Current resin inventory and open orders
        current_resin_inventory = self.safe_numeric_convert(row.get('Current_Resin_Inventory_kg', 0))
        resin_open_orders = self.safe_numeric_convert(row.get('Resin_Open_Order_kg', 0))
        available_resin = current_resin_inventory + resin_open_orders

        # Safety stock (based on NET demand variability and lead time)
        safety_stock = self.calculate_safety_stock_with_leadtime(weekly_net_demands, lead_time_weeks)

        # Reorder Point = Demand during lead time + Safety Stock
        reorder_point = demand_during_leadtime + safety_stock

        # Order Quantity = max(0, Reorder Point - Available Resin)
        # Also ensure we cover at least the monthly NET demand
        min_order_qty = max(0, monthly_net_demand_kg - available_resin)
        order_qty_from_rop = max(0, reorder_point - available_resin)
        order_qty = max(min_order_qty, order_qty_from_rop)

        # Projected ending inventory
        projected_ending = available_resin + order_qty - monthly_net_demand_kg

        # Stock coverage in weeks
        stock_coverage_weeks = available_resin / avg_weekly_net_demand if avg_weekly_net_demand > 0 else float('inf')

        # Determine urgency
        if monthly_net_demand_kg == 0:
            urgency = "NO DEMAND - No Production Needed"
        elif stock_coverage_weeks < lead_time_weeks:
            urgency = "URGENT - Order Immediately"
        elif stock_coverage_weeks < lead_time_weeks * 1.5:
            urgency = "HIGH - Order Soon"
        elif order_qty > 0:
            urgency = "NORMAL - Plan Order"
        else:
            urgency = "LOW - Sufficient Stock"

        return {
            'lead_time_weeks': lead_time_weeks,
            'weekly_net_demands_kg': weekly_net_demands,
            'avg_weekly_net_demand_kg': avg_weekly_net_demand,
            'monthly_net_demand_kg': monthly_net_demand_kg,
            'demand_during_leadtime_kg': demand_during_leadtime,
            'current_resin_inventory_kg': current_resin_inventory,
            'resin_open_orders_kg': resin_open_orders,
            'available_resin_kg': available_resin,
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
        2. Calculate NET demand at FG level (Gross Demand - FG Inventory)
        3. Aggregate NET demand to Resin PartNo level
        4. Calculate ordering recommendations with lead time consideration
        """
        self.log(f"Processing resin inventory from: {input_file_path}")

        # Read the BOM data
        df = self.read_resin_bom(input_file_path)

        # Identify demand columns
        demand_cols = self.identify_demand_columns(df)
        num_weeks = len(demand_cols)
        self.log(f"Found {num_weeks} demand forecast columns: {demand_cols}")

        # Step 1: Calculate NET demand at FG level
        df_fg = self.calculate_net_demand_at_fg_level(df, demand_cols)

        # Step 2: Aggregate NET demand at Resin level
        aggregated = self.aggregate_at_resin_level(df_fg, demand_cols)

        # Step 3: Calculate ordering recommendations
        self.log("Step 3: Calculating ordering recommendations with lead time...")
        recommendations = []

        for idx, row in aggregated.iterrows():
            calc = self.calculate_ordering_recommendation(row, num_weeks)

            rec = {
                'Resin_PartNo': row['Resin_PartNo'],
                'Num_FGs_Using': row['Num_FGs_Using_Resin'],
                'Customers': row['Customer'],
                'Models': row['Model'],
                'Lead_Time_Weeks': calc['lead_time_weeks'],
                'Avg_Weight_per_Unit_kg': round(row['Weight_kg'], 6),
                # Gross vs Net comparison
                'Gross_Demand_Units': round(row['Total_Demand_Units'], 0),
                'FG_Inventory_Units': round(row['FG_Inventory_Units'], 0),
                'Net_Demand_Units': round(row['Net_Demand_Units'], 0),
                # Weekly NET demands in kg
                'Week1_Net_Demand_kg': round(calc['weekly_net_demands_kg'][0], 3) if len(
                    calc['weekly_net_demands_kg']) > 0 else 0,
                'Week2_Net_Demand_kg': round(calc['weekly_net_demands_kg'][1], 3) if len(
                    calc['weekly_net_demands_kg']) > 1 else 0,
                'Week3_Net_Demand_kg': round(calc['weekly_net_demands_kg'][2], 3) if len(
                    calc['weekly_net_demands_kg']) > 2 else 0,
                'Week4_Net_Demand_kg': round(calc['weekly_net_demands_kg'][3], 3) if len(
                    calc['weekly_net_demands_kg']) > 3 else 0,
                # Summary metrics
                'Avg_Weekly_Net_Demand_kg': round(calc['avg_weekly_net_demand_kg'], 3),
                'Monthly_Net_Demand_kg': round(calc['monthly_net_demand_kg'], 3),
                'Demand_During_LeadTime_kg': round(calc['demand_during_leadtime_kg'], 3),
                # Resin inventory
                'Current_Resin_Inventory_kg': round(calc['current_resin_inventory_kg'], 3),
                'Resin_Open_Orders_kg': round(calc['resin_open_orders_kg'], 3),
                'Available_Resin_kg': round(calc['available_resin_kg'], 3),
                # Ordering calculation
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
            'LOW - Sufficient Stock': 3,
            'NO DEMAND - No Production Needed': 4
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
        print("\n" + "=" * 95)
        print("RESIN INVENTORY ORDERING SUMMARY REPORT v3.0")
        print("(with FG Inventory Deduction and Lead Time Analysis)")
        print("=" * 95)

        print(f"\n📊 OVERVIEW:")
        print(f"   Total unique Resin PartNo: {len(output_df)}")

        # Demand reduction summary
        total_gross = output_df['Gross_Demand_Units'].sum()
        total_fg_inv = output_df['FG_Inventory_Units'].sum()
        total_net = output_df['Net_Demand_Units'].sum()
        reduction_pct = (1 - total_net / total_gross) * 100 if total_gross > 0 else 0

        print(f"\n   📦 FG Inventory Impact:")
        print(f"      Gross Demand: {total_gross:,.0f} units")
        print(f"      FG Inventory on Hand: {total_fg_inv:,.0f} units")
        print(f"      NET Demand (needs production): {total_net:,.0f} units")
        print(f"      Demand Reduction: {reduction_pct:.1f}%")

        # Count by urgency
        urgency_counts = output_df['Order_Urgency'].value_counts()
        print(f"\n   🚦 Order Urgency Breakdown:")
        for urgency, count in urgency_counts.items():
            print(f"      {urgency}: {count}")

        total_net_demand_kg = output_df['Monthly_Net_Demand_kg'].sum()
        total_order = output_df['Recommended_Order_kg'].sum()
        print(f"\n   📊 Resin Requirements:")
        print(f"      Total NET monthly demand: {total_net_demand_kg:,.2f} kg")
        print(f"      Total recommended order: {total_order:,.2f} kg")

        # Lead time distribution
        print("\n   ⏱️ Lead Time Distribution:")
        lt_dist = output_df['Lead_Time_Weeks'].value_counts().sort_index()
        for lt, count in lt_dist.items():
            print(f"      {lt} weeks: {count} resins")

        print("\n" + "-" * 95)
        print("🔴 URGENT ORDERS (Stock Coverage < Lead Time):")
        print("-" * 95)
        urgent = output_df[output_df['Order_Urgency'] == 'URGENT - Order Immediately']
        if len(urgent) > 0:
            for _, row in urgent.head(10).iterrows():
                resin_name = row['Resin_PartNo'][:42] + "..." if len(str(row['Resin_PartNo'])) > 42 else row[
                    'Resin_PartNo']
                print(f"   {resin_name}")
                print(
                    f"      Order: {row['Recommended_Order_kg']:.1f} kg | Net Demand: {row['Monthly_Net_Demand_kg']:.1f} kg | LT: {row['Lead_Time_Weeks']} wks | Coverage: {row['Stock_Coverage_Weeks']} wks")
            if len(urgent) > 10:
                print(f"   ... and {len(urgent) - 10} more urgent items")
        else:
            print("   No urgent orders!")

        print("\n" + "-" * 95)
        print("🟢 NO PRODUCTION NEEDED (FG Inventory covers demand):")
        print("-" * 95)
        no_demand = output_df[output_df['Order_Urgency'] == 'NO DEMAND - No Production Needed']
        if len(no_demand) > 0:
            for _, row in no_demand.iterrows():
                resin_name = row['Resin_PartNo'][:42] + "..." if len(str(row['Resin_PartNo'])) > 42 else row[
                    'Resin_PartNo']
                print(f"   {resin_name}")
                print(
                    f"      Gross Demand: {row['Gross_Demand_Units']:.0f} units | FG Inventory: {row['FG_Inventory_Units']:.0f} units | Surplus: {row['FG_Inventory_Units'] - row['Gross_Demand_Units']:.0f} units")
        else:
            print("   All resins have production requirements")

        print("\n" + "-" * 95)
        print("📋 TOP 10 BY ORDER QUANTITY:")
        print("-" * 95)
        top_orders = output_df[output_df['Recommended_Order_kg'] > 0].nlargest(10, 'Recommended_Order_kg')
        for _, row in top_orders.iterrows():
            resin_name = row['Resin_PartNo'][:42] + "..." if len(str(row['Resin_PartNo'])) > 42 else row['Resin_PartNo']
            print(f"   {resin_name}")
            print(
                f"      Order: {row['Recommended_Order_kg']:.1f} kg | Net Demand: {row['Monthly_Net_Demand_kg']:.1f} kg | LT: {row['Lead_Time_Weeks']} wks")

        print("\n" + "=" * 95)
        return output_df


def main():
    """Main entry point"""
    import sys

    manager = ResinInventoryManager(verbose=True)

    # Default input file path
    input_file = "Resin_BOM_list_test_new.xlsx"
    output_file = "resin_ordering_recommendations_v3.csv"

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