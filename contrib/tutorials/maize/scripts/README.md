Steps
=====

## Filter CSV
- Script: `filter_top_poses.py stage2_1.csv top.sdf`
- Various filter criteria and write to `top.csv`
- Extract top poses for subset

## Run Gnina local optimization
- Script: `dock_local.sh adgpu_prepare/1DB5.pdbqt top.sdf gnina_local.sdf > gnina_local.log`
- Local optimization of poses in `top.sdf` docked in `1DB5.pdbqt`

## Get best Gnina poses
- Script: `get_best_local.py gnina_local.sdf best_pose_local.sdf`
- All poses with `minimizedRMSD` < 0.5 and highest `CNNaffinity` for each molecule
