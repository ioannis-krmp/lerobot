#!/bin/bash

# Create assets directory
mkdir -p /home/TODO/lerobot/examples/so100_to_so100_EE/SO101/assets
cd /home/TODO/lerobot/examples/so100_to_so100_EE/SO101/assets

# Base URL for mesh files
BASE_URL="https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/assets"

# List of mesh files to download
mesh_files=(
    "base_so101_v2.stl"
    "sts3215_03a_v1.stl"
    "waveshare_mounting_plate_so101_v2.stl"
    "motor_holder_so101_base_v1.stl"
    "rotation_pitch_so101_v1.stl"
    "upper_arm_so101_v1.stl"
    "under_arm_so101_v1.stl"
    "motor_holder_so101_wrist_v1.stl"
    "sts3215_03a_no_horn_v1.stl"
    "wrist_roll_pitch_so101_v2.stl"
    "wrist_roll_follower_so101_v1.stl"
    "moving_jaw_so101_v1.stl"
)

# Download each file
for file in "${mesh_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Downloading $file..."
        wget "$BASE_URL/$file"
    else
        echo "$file already exists, skipping..."
    fi
done

echo "All mesh files downloaded successfully!"