import numpy as np
import matplotlib.pyplot as plt
import copy

import evo.tools.file_interface as file_interface
import evo.core.sync as sync
import evo.core.metrics as metrics

# ==========================
# FILE PATHS
# ==========================
gt_file = "/home/vm/TUM/groundtruth.txt"
est_file = "/home/vm/ORB_SLAM2/CameraTrajectory.txt"

# ==========================
# LOAD TRAJECTORIES
# ==========================
traj_gt = file_interface.read_tum_trajectory_file(gt_file)
traj_est = file_interface.read_tum_trajectory_file(est_file)

# ==========================
# SYNCHRONIZE TIMESTAMPS
# (added tolerance for safety)
# ==========================
traj_gt, traj_est = sync.associate_trajectories(
    traj_gt, traj_est, max_diff=0.02
)

# Safety check
if len(traj_gt.timestamps) == 0:
    raise ValueError("❌ No matching timestamps found. Check dataset and trajectory!")

# ==========================
# ALIGN TRAJECTORIES
# ==========================
traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_gt, correct_scale=True)

# ==========================
# COMPUTE ATE
# ==========================
ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
ate_metric.process_data((traj_gt, traj_est_aligned))
ate_stats = ate_metric.get_all_statistics()

print("\n===== ATE RESULTS =====")
for k, v in ate_stats.items():
    print(f"{k:>10}: {v:.6f}")

# ==========================
# COMPUTE RPE
# ==========================
rpe_metric = metrics.RPE(
    metrics.PoseRelation.translation_part,
    delta=1,
    delta_unit=metrics.Unit.frames
)

rpe_metric.process_data((traj_gt, traj_est_aligned))
rpe_stats = rpe_metric.get_all_statistics()

print("\n===== RPE RESULTS =====")
for k, v in rpe_stats.items():
    print(f"{k:>10}: {v:.6f}")

# ==========================
# PLOT TRAJECTORY (Top View)
# ==========================
plt.figure()
plt.plot(
    traj_gt.positions_xyz[:, 0],
    traj_gt.positions_xyz[:, 2],
    label="Ground Truth"
)
plt.plot(
    traj_est_aligned.positions_xyz[:, 0],
    traj_est_aligned.positions_xyz[:, 2],
    label="Estimated"
)
plt.legend()
plt.title("Trajectory Comparison (Top View X-Z)")
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.grid()

# ==========================
# PLOT ATE ERROR
# ==========================
plt.figure()
plt.plot(ate_metric.error)
plt.title("ATE Error Over Time")
plt.xlabel("Frame")
plt.ylabel("Error (m)")
plt.grid()

# ==========================
# PLOT RPE ERROR
# ==========================
plt.figure()
plt.plot(rpe_metric.error)
plt.title("RPE Error Over Time")
plt.xlabel("Frame")
plt.ylabel("Error (m)")
plt.grid()

# ==========================
# SAVE RESULTS (IMPORTANT FOR REPORT)
# ==========================
np.savetxt("ate_errors.txt", ate_metric.error)
np.savetxt("rpe_errors.txt", rpe_metric.error)

plt.savefig("trajectory_plot.png")  # saves last figure

plt.show()
