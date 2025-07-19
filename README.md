# Retail-Analytics-Suite

- [Python]
- [Pandas] 
- [MLxtend] 

A comprehensive retail analytics solution featuring synthetic data generation, top-selling product identification, and market basket analysis using Apriori algorithm.

## Features

### 🛒 Data Generation (`generator.py`)
- Generates realistic retail transactions with:
  - Multi-item purchases (2-5 items per transaction)
  - Built-in product correlations (e.g., Burrito + Taco)
  - Configurable date ranges and transaction volumes
  - Output formats: CSV or Parquet

### 📈 Core Analytics (`main.py`)
- **Top Product Identification**:
  - Ranks products by sales volume (amount)
  - Provides item metadata (name, category)

- **Market Basket Analysis**:
  - Apriori algorithm for association rule mining
  - Confidence metrics for product pairings
  - Filters results to top-selling items only

### 📂 Data Assets
- `items.csv`: Product catalog with 50+ food/beverage items
- `transactions.parquet`: Sample generated transaction data

## 🚀 How to Run

- Create and activate the virtual environment (if it doesn't exist)
  - python -m venv venv
  - .\venv\Scripts\activate
- Install dependencies
  - pip install -r requirements.txt
- Generate sample data
  -  python generator.py --start_date 2023-01-01 --end_date 2024-12-31 --rows_per_year 500000 --output_format parquet
- Run the main script
  - python main.py --target_date (date)
  - Note: Replace (date) with the desired date in YYYY-MM-DD format  
  - Example: python main.py 
