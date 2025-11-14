# NeRF Studio extract frames only
ns-process-data video --data /path/to/your/video.mp4 --output-dir . --skip-colmap


# 1. Feature extraction with GPU
colmap feature_extractor \
  --database_path colmap/database.db \
  --image_path videos/images \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1

# 2. Sequential matching with GPU (for video data)
colmap sequential_matcher \
  --database_path colmap/database.db \
  --FeatureMatching.use_gpu 1 \
  --SequentialMatching.overlap 10

# 3. Clean up old sparse reconstructions
rm -rf colmap/sparse/*

# 4. Sparse reconstruction with GPU bundle adjustment
colmap mapper \
  --database_path colmap/database.db \
  --image_path images \
  --output_path colmap/sparse \
  --Mapper.ba_use_gpu 1

# 5. Check for largest directory in case the reconstruction is disconnected
for dir in colmap/sparse/*/; do 
  echo -n "$dir: "
  colmap model_analyzer --path "$dir" 2>&1 | grep "Registered images" || echo "N/A"
done

# 6. Convert to text format for NeRF Studio (optional - NeRF Studio can read binary files)
colmap model_converter \
  --input_path colmap/sparse/0 \
  --output_path colmap/sparse/0 \
  --output_type TXT

# NeRF Studio Training

# Train Gaussian Splatting (Fast & High Quality)
ns-train splatfacto colmap --data . --colmap-path colmap/sparse/9

# Train Gaussian Splatting (for large/outdoor scenes)
ns-train splatfacto-big colmap --data . --colmap-path colmap/sparse/9

# Train NeRF (Classic NeRF approach)
ns-train nerfacto colmap --data . --colmap-path colmap/sparse/9

# Note: Replace "9" with your largest sparse reconstruction folder number


# PyTorch 2.6 Compatibility Fix
# If you get pickle errors when loading checkpoints, apply this fix:
# Edit: venv/lib/python3.12/site-packages/nerfstudio/utils/eval_utils.py
# Line 62: Change
#   loaded_state = torch.load(load_path, map_location="cpu")
# To:
#   loaded_state = torch.load(load_path, map_location="cpu", weights_only=False)


# Viewing Results

# Interactive viewer (navigate in real-time)
ns-viewer --load-config outputs/unnamed/splatfacto/2025-11-13_172603/config.yml

# Rendering Videos

# 1. Render interpolated path (automatic flythrough)
ns-render interpolate --load-config outputs/unnamed/splatfacto/2025-11-13_172603/config.yml \
  --output-path renders/output.mp4 \
  --frame-rate 30

# 2. Render custom camera path (create in viewer first)
ns-render camera-path --load-config outputs/unnamed/splatfacto/2025-11-13_172603/config.yml \
  --camera-path-filename camera_path.json \
  --output-path renders/custom.mp4

# 3. Render original training views
ns-render dataset --load-config outputs/unnamed/splatfacto/2025-11-13_172603/config.yml \
  --output-path renders/frames/ \
  --split train

# Exporting Gaussian Splats

# Export 3D Gaussian Splat as .ply file
ns-export gaussian-splat --load-config outputs/unnamed/splatfacto/2025-11-13_172603/config.yml \
  --output-dir exports/splat/

# View exported .ply in:
# - SuperSplat: https://playcanvas.com/supersplat/editor
# - WebGL Viewer: https://antimatter15.com/splat/
# - Local: gsplat-viewer exports/splat/splat.ply
