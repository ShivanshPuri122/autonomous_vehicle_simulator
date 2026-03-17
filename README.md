# Autonomous Vehicle Simulator

CARLA-based autonomous driving pipeline with lane detection and object detection.

## Project Structure
```
autonomous_vehicle_simulator/
├── config/         - YAML configs (CARLA settings, dataset configs)
├── data/           - Datasets (NOT in git - shared via Google Drive)
├── models/         - Trained weights (NOT in git - shared via Google Drive)
├── perception/     - Lane detection + Object detection
├── sensors/        - RGB camera + LiDAR setup
├── controls/       - PID controller
├── simulation/     - CARLA client + environment
├── scripts/        - Dataset preparation tools
├── notebooks/      - Google Colab training notebooks
└── tests/          - Unit tests