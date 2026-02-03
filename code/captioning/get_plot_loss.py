import json
import os
import numpy as np
import matplotlib.pyplot as plt

FOLDER = "path"
STATE_FILE = os.path.join(FOLDER, "trainer_state.json")
INCLUDE_FIRST_VAL = False
JUST_VAL = True
k = 3  # smoothing window

def smooth_curve(values, window):
    values = np.array(values)
    if len(values) <= window:
        return values  # no smoothing if not enough samples
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(values[start:i+1].mean())
    return np.array(smoothed)

with open(STATE_FILE, "r") as f:
    state = json.load(f)

log_history = state["log_history"]

train_steps = []
train_loss = []

val_steps = []
val_loss = []

# parse
for entry in log_history:
    if "loss" in entry and "learning_rate" in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])

    if "eval_loss" in entry:
        val_steps.append(entry["step"])
        val_loss.append(entry["eval_loss"])

# first validation step
first_val_step = val_steps[0]

# filter training from first validation
train_steps_f = []
train_loss_f = []

for step, loss in zip(train_steps, train_loss):
    if step >= first_val_step:
        train_steps_f.append(step)
        train_loss_f.append(loss)

# apply smoothing
train_loss_s = smooth_curve(train_loss_f, k)
val_loss_s = smooth_curve(val_loss, k)

# plot
plt.figure(figsize=(10, 6))
if not JUST_VAL:
    plt.plot(train_steps_f, train_loss_s, label="Train Loss (smoothed)")
plt.plot(val_steps, val_loss_s, label="Validation Loss (smoothed)")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title(f"Training and Validation Loss with {k}-point Smoothing")
plt.legend()

out_path = os.path.join(FOLDER, "loss_plot_smoothed.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved plot to:", out_path)
