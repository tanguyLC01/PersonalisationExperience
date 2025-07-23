import itertools
import subprocess

# Define valid levels (0–14)
levels = list(range(5))

# Generate combinations for 3 clients
combinations = [
    combo for combo in itertools.product(levels, repeat=3)
    if not (all(x == 0 for x in combo) or combo == (0, 0, 1))
]

print(f"Running {len(combinations)} combinations...")

for i, combo in enumerate(combinations):
    combo_str = ",".join(map(str, combo))
    cmd = ["python", "run.py", f"model.personalisation_level=[{combo_str}]"]

    print(f"[{i+1}/{len(combinations)}] Running: {cmd}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed for {combo}: {e}")
