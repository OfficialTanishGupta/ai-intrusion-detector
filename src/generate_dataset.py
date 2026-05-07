import pandas as pd
import random

rows = []

protocols = ['tcp', 'udp', 'icmp']
flags = ['SF', 'S0', 'REJ']

# Generate NORMAL traffic
for _ in range(500):
    duration = random.randint(1, 20)
    protocol = random.choice(protocols)

    src_bytes = random.randint(100, 3000)
    dst_bytes = random.randint(100, 3000)

    flag = 'SF'

    label = 0

    rows.append([
        duration,
        protocol,
        src_bytes,
        dst_bytes,
        flag,
        label
    ])

# Generate ATTACK traffic
for _ in range(500):
    duration = random.randint(20, 100)

    protocol = random.choice(protocols)

    src_bytes = random.randint(3000, 10000)
    dst_bytes = random.randint(0, 500)

    flag = random.choice(['REJ', 'S0'])

    label = 1

    rows.append([
        duration,
        protocol,
        src_bytes,
        dst_bytes,
        flag,
        label
    ])

# Create dataframe
df = pd.DataFrame(rows, columns=[
    'duration',
    'protocol_type',
    'src_bytes',
    'dst_bytes',
    'flag',
    'label'
])

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save CSV
df.to_csv('../data/network_data.csv', index=False)

print("✅ Dataset generated successfully")
print(df.head())