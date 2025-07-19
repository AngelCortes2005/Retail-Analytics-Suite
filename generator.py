import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse


# Function to generate random dates
def random_dates(start, end, n):
    date_range = (end - start).days
    random_days = np.random.randint(0, date_range, n)
    return [start + timedelta(days=int(day)) for day in random_days]


# Generate correlated item sets for multi-item transactions
def generate_correlated_items(num_multi_item_transactions, multi_items_per_transaction):
    correlated_items = []

    for i in range(num_multi_item_transactions):
        transaction_items = set()

        # Introduce correlation: 20% of the transactions will include item_id 1 and 2 together
        if np.random.rand() < 0.2:
            transaction_items.update([1, 2])

        # Randomly generate the remaining items
        while len(transaction_items) < multi_items_per_transaction[i]:
            transaction_items.add(np.random.randint(1, 11))

        correlated_items.append(list(transaction_items))

    return correlated_items


# Generate data
def generate_data(rows_per_year, start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate the number of multi-item and single-item transactions
    num_multi_item_transactions = int(
        rows_per_year * 0.85 / 4
    )  # 85% of rows, 4 items on avg per transaction
    num_single_item = rows_per_year - (num_multi_item_transactions * 4)

    # Multi-item transactions will have between 2 and 5 items
    multi_items_per_transaction = np.random.randint(
        2, 6, size=num_multi_item_transactions
    )

    # Generate random times
    random_seconds = np.random.randint(0, 86400, rows_per_year)  # Seconds in a day
    times = pd.to_datetime(random_seconds, unit="s").time  # Extract time component

    # Generate correlated items for multi-item transactions
    correlated_item_sets = generate_correlated_items(
        num_multi_item_transactions, multi_items_per_transaction
    )

    # Multi-item transactions
    multi_rows = np.sum(
        multi_items_per_transaction
    )  # Total rows needed for multi-item transactions

    data_multi = {
        "id": np.repeat(
            np.arange(1, num_multi_item_transactions + 1), multi_items_per_transaction
        ),
        "pos_code": np.random.randint(1, 1000, size=multi_rows),  # POS code range
        "store_id": np.random.randint(
            1, 16, size=multi_rows
        ),  # Store ID between 1 and 15
        "channel_id": np.random.randint(
            1, 4, size=multi_rows
        ),  # Channel ID between 1 and 3
        "item_id": np.hstack(correlated_item_sets),  # Use correlated item sets
        "amount": np.random.randint(1500, 12000, size=multi_rows),
        "date": random_dates(start, end, multi_rows),  # Random dates
        "time": times[:multi_rows],  # Random time
    }

    # Generate single-item transactions
    data_single = {
        "id": np.arange(
            num_multi_item_transactions + 1,
            num_multi_item_transactions + num_single_item + 1,
        ),
        "pos_code": np.random.randint(1, 1000, size=num_single_item),  # POS code range
        "store_id": np.random.randint(
            1, 16, size=num_single_item
        ),  # Store ID between 1 and 15
        "channel_id": np.random.randint(
            1, 4, size=num_single_item
        ),  # Channel ID between 1 and 3
        "item_id": np.random.randint(
            1, 11, size=num_single_item
        ),  # Item ID between 1 and 50
        "amount": np.random.randint(1500, 12000, size=num_single_item),
        "date": random_dates(start, end, num_single_item),  # Random dates
        "time": times[multi_rows : multi_rows + num_single_item],  # Random time
    }

    # Combine multi-item and single-item transactions
    df_multi = pd.DataFrame(data_multi)
    df_single = pd.DataFrame(data_single)

    # Concatenate both DataFrames
    df = pd.concat([df_multi, df_single])

    # Shuffle the rows to mix single and multi-item transactions
    df = df.sample(frac=1).reset_index(drop=True)

    return df


# Main function to handle the argument parsing
def main():
    parser = argparse.ArgumentParser(description="Generate mock transactional data.")

    # Add arguments
    parser.add_argument(
        "--start_date", required=True, help="The start date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end_date", required=True, help="The end date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--rows_per_year",
        type=int,
        default=500000,
        help="Number of rows to generate per year (default: 500,000).",
    )
    parser.add_argument(
        "--output_format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format: csv or parquet (default: csv).",
    )
    parser.add_argument(
        "--output_file",
        default="transactions",
        help="The name of the output file (without extension).",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Generate the data
    df = generate_data(args.rows_per_year, args.start_date, args.end_date)

    # Save to the chosen format
    if args.output_format == "csv":
        df.to_csv(f"{args.output_file}.csv", index=False)
    elif args.output_format == "parquet":
        df.to_parquet(f"{args.output_file}.parquet", index=False)

    print(f"Data generation complete! Saved to {args.output_file}.{args.output_format}")


# Example way to call this file:
# python generator.py --start_date 2023-01-01 --end_date 2024-09-25 --rows_per_year 2500000 --output_format parquet --output_file transactions
if __name__ == "__main__":
    main()
