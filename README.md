# DroneVision Project

This repository contains preliminary experiments for running YOLO models with OpenCV and ONNX Runtime. The current layout mixes build artifacts and source files in the top level. The suggestions below provide a cleaner organization for future development.

## Suggested Folder Layout

```
DroneVision/
├── CMakeLists.txt
├── src/            # C++ source files
├── include/        # C++ header files (if any)
├── models/         # Pretrained model files
├── data/           # Example images or videos
├── scripts/        # Utility scripts for setup or dataset preparation
├── tests/          # Test cases
├── build/          # CMake build output (should be git‑ignored)
└── docs/           # Project documentation
```

Key points:

- Keep build artifacts out of version control by adding `build/` to `.gitignore`.
- Place application code inside `src/` and headers in `include/` for clarity.
- Use `models/` or `data/` to store external assets rather than mixing them with source files.
- Document design decisions and usage instructions in `docs/`.

These changes make it easier to navigate the repository and separate generated files from source code.
