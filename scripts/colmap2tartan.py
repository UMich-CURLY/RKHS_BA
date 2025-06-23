import sys, os
import numpy as np

def colmap_to_tartanair(colmap_file, output_file):
    poses = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
    with open(colmap_file, 'r') as f:
        lines = f.readlines()
    
    # Parse COLMAP's images.txt (skip comments and empty lines)
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if not parts[-1].lower().endswith(image_extensions):
            continue  # Skip lines without image extensions

        # Extract QW, QX, QY, QZ, TX, TY, TZ (COLMAP format)
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        
        # Reorder to TartanAir format (x y z qx qy qz qw)
        poses.append(f"{tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
    
    # Save to output file
    with open(output_file, 'w') as f:
        f.writelines(poses)

# Example usage
colmap_to_tartanair(sys.argv[1], sys.argv[2])
