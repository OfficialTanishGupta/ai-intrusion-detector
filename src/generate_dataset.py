import pandas as pd
import random

rows = []

protocols = ['tcp', 'udp', 'icmp']


for _ in range(500):
    rows.append([
        random.randint(1, 20),
        random.choice(protocols),
        random.randint(100, 3000),
        random.randint(100, 3000),
        'SF',
        0 # normal
    ])


for _ in range(300):
    rows.append([
        random.randint(50, 100),
        'tcp',
        random.randint(5000, 10000),
        random.randint(0, 500),
        'S0',
        1 # dos
    ])

for _ in range(200):
    rows.append([
        random.randint(20, 50),
        'icmp',
        random.randint(1000, 4000),
        random.randint(50, 1000),
        'REJ',
        2 # probe
    ])


for _ in range(150):
    rows.append([
        random.randint(5, 30),
        'udp',
        random.randint(500, 2000),
        random.randint(100, 500),
        'SF',
        3 # r2l
    ])


for _ in range(100):
    rows.append([
        random.randint(10, 40),
        'tcp',
        random.randint(2000, 6000),
        random.randint(0, 300),
        'REJ',
        4 # u2r
    ])

df = pd.DataFrame(rows, columns=[
    'duration',
    'protocol_type',
    'src_bytes',
    'dst_bytes',
    'flag',
    'label'
])

df = df.sample(frac=1).reset_index(drop=True)

# Save CSV
df.to_csv('../data/network_data.csv', index=False)

print("✅ Multi-class dataset generated successfully!")
print(f"Total records: {len(df)}")
print(df.head())
