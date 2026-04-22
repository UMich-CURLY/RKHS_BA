import sys, os, numpy

def g2o_to_tartanair(g2o_file, output_file):
    poses = []
    
    with open(g2o_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if not line.startswith("VERTEX_SE3:QUAT"):
            continue  # Skip non-pose lines (e.g., edges, comments)
        
        parts = line.strip().split()
        if len(parts) < 8:
            continue  # Ensure valid pose entry
        
        # Extract ID, x, y, z, qx, qy, qz, qw (g2o format)
        x, y, z, qx, qy, qz, qw = parts[2:]
        
        # Write in TartanAir format (x y z qx qy qz qw)
        poses.append(f"{x} {y} {z} {qx} {qy} {qz} {qw}\n")
    
    with open(output_file, 'w') as f:
        f.writelines(poses)

# Example usage
g2o_to_tartanair(sys.argv[1], sys.argv[2])
