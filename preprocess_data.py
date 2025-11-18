import json

with open("./data/industry_kline_data.json", "r", encoding="utf-8") as f:
    source_data = json.loads(f.read())
for k,v in source_data.items():
    print(k, v[0][0], v[len(v)-1][0])