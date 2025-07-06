# Folder Structure Proposal

The repository currently contains build artifacts under `build/` and multiple C++ files at the top level. To make the project easier to maintain, consider the following structure:

```
DroneVision/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   └── ...
├── include/
├── models/
├── data/
├── scripts/
├── tests/
├── docs/
└── build/    # Excluded from Git
```

- **src/** – All application source files.
- **include/** – Header files if the project grows.
- **models/** – Pretrained models (e.g., ONNX files).
- **data/** – Sample input images or video streams.
- **scripts/** – Helper scripts for environment setup or dataset processing.
- **tests/** – Unit or integration tests.
- **docs/** – Documentation like this file.
- **build/** – Compiled output; should be listed in `.gitignore` so it is not checked in.

This layout keeps version-controlled files small and clearly separates code from generated artifacts.

See also `main_cpp_overview.md` for a description of the application logic and
`library_overview.md` for details about the helper library.
