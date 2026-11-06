# downtown-kit

Lightweight geospatial pipeline library for building multi-LOD, metric glTF 2.0 digital twins and game engine environments.

Repo: https://github.com/wolfgangs-code/downtown-kit

## Features
- **Geodesy & Local Frame**: Projections (WGS84, State Plane, UTM), street grid rotation, local-to-glTF $(X, Z, -Y)$ metric conversion.
- **Elevation**: USGS 3DEP LiDAR 1m & regional elevation ingestion with GDAL warping and void filling.
- **Vector Infrastructure**: Overpass API streaming with multi-mirror fallback and multipolygon courtyard resolution.
- **Aerial Imagery**: Bing Maps Quadkey tile fetching, parallel streaming, and georeferenced GeoTIFF creation.
- **Mesh Optimization**: Hyperplane observer bounding-box backface culling.
- **Export**: PBR materials and metric glTF 2.0 binary (`.glb`) packaging.

## Installation in Colab
```bash
!pip install -q git+https://github.com/wolfgangs-code/downtown-kit.git
```

Or for live local development:
```python
!git clone https://github.com/wolfgangs-code/downtown-kit.git
import sys; sys.path.append("/content/downtown-kit")
```
