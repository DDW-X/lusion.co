#!/usr/bin/env bash
# Bash GitHub Metadata Configuration Script
# Requires: GitHub CLI (gh) authenticated (gh auth login)

set -euo pipefail

echo "Configuring GitHub Repository Metadata for lusion.co..."

# 1. Update About Description and Homepage
gh repo edit --description "Comprehensive reverse engineering & mathematical deconstruction of lusion.co: Custom GLSL Liquid Glass, GPGPU Curl Particles, V8 Zero-GC Engine, 4D Möbius Conformal Geometry & 120 FPS Systems Architecture. Authored by DDW-X." --homepage "https://lusion.co"

# 2. Inject Maximum Visibility Topics
topics=(
    "reverse-engineering"
    "webgl"
    "webgl2"
    "webgpu"
    "glsl"
    "threejs"
    "gpgpu"
    "shaders"
    "computer-graphics"
    "creative-coding"
    "v8-engine"
    "performance-optimization"
    "zero-allocation"
    "fluid-dynamics"
    "navier-stokes"
    "physics-engine"
    "non-euclidean-geometry"
    "web-audio-api"
    "draco-compression"
    "systems-architecture"
)

for topic in "${topics[@]}"; do
    echo "Adding topic: $topic"
    gh repo edit --add-topic "$topic"
done

echo "GitHub Repository Metadata successfully configured!"
