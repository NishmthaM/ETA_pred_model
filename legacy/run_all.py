
import subprocess
import sys

# Run the files in the correct order

files = [
    "synthetic.py",
    "train_xgboost.py",
    "live_prediction.py"
]

for file in files:

    print(f"\nRunning {file}...")

    result = subprocess.run(
        [sys.executable, file]
    )

    if result.returncode != 0:
        print(f"Error while running {file}")
        sys.exit(result.returncode)

print("\nAll files executed successfully!")