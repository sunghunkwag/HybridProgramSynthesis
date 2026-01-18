
import os
import json
import glob

def get_latest_metrics():
    # Find all metrics.jsonl files
    files = glob.glob("results/**/metrics.jsonl", recursive=True)
    if not files:
        print("No metrics.jsonl found.")
        return

    # Sort by modification time
    latest_file = max(files, key=os.path.getmtime)
    print(f"Reading latest log: {latest_file}")
    
    last_line = ""
    with open(latest_file, 'r') as f:
        for line in f:
            if line.strip():
                last_line = line
    
    if last_line:
        try:
            data = json.loads(last_line)
            print(json.dumps(data, indent=2))
        except:
            print(f"Raw Line: {last_line}")
    else:
        print("Log file is empty.")

if __name__ == "__main__":
    get_latest_metrics()
