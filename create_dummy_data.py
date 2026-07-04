"""Generates a realistic messy sales dataset for pipeline testing."""
import random
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)

PRODUCTS = ["Laptop", "Monitor", "Keyboard", "Mouse", "Headset", "Webcam", "Desk Chair", "USB Hub"]
CATEGORIES = ["Electronics", "electronics", "ELECTRONICS", "Furniture", "furniture"]
REGIONS = ["North", "South", "East", "West", "north", "WEST"]
REPS = ["Alice Johnson", "Bob Smith", "Carol White", "David Lee", "Eve Martinez"]
STATUSES = ["Completed", "completed", "Pending", "PENDING", "Cancelled", "completed"]

def random_date():
    start = datetime(2023, 1, 1)
    return (start + timedelta(days=random.randint(0, 730))).strftime("%Y-%m-%d")

rows = []
for i in range(1, 201):
    qty = random.randint(1, 50)
    price = round(random.uniform(20, 2000), 2)
    # inject outliers
    if i in [15, 47, 89, 134, 178]:
        qty = random.randint(500, 1000)
        price = round(random.uniform(50000, 100000), 2)

    rows.append({
        "Customer ID ": f"CUST-{i:04d}",           # trailing space in header
        "Customer  Name": random.choice(["John Doe", "Jane Smith", "Acme Corp", "Beta LLC", None if i % 20 == 0 else "Global Inc"]),
        "Sale Date": random_date(),
        "Product Name": random.choice(PRODUCTS),
        "Category": random.choice(CATEGORIES),
        "Qty Ordered": qty,
        "Unit Price ($)": price,                    # special chars in header
        "Total Revenue": round(qty * price, 2) if i % 30 != 0 else None,  # some nulls
        "Region/Territory": random.choice(REGIONS), # slash in header
        "Sales Rep": random.choice(REPS),
        "Deal Status": random.choice(STATUSES),
        "Discount%": round(random.uniform(0, 0.3), 2) if i % 5 != 0 else None,
        "Notes": None if i % 3 == 0 else f"Note {i}",
    })

# add duplicate rows
rows.extend(rows[10:15])

df = pd.DataFrame(rows)
df.to_excel("data/samples/sales_data.xlsx", index=False)
print(f"Created data/samples/sales_data.xlsx — {len(df)} rows, {len(df.columns)} columns")
print(f"Nulls: {df.isnull().sum().to_dict()}")
