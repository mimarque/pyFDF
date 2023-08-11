# PyFDF

PyFDF is a Python program for visualizing 3D wireframe models from FDF files. It uses PyQt5 for the graphical interface and provides various projections like orthographic, isometric, and perspective, along with rotation and scaling options.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Mathematics and Implementation](#Mathematics-and-Implementation)

## Introduction

PyFDF is a 3D wireframe visualization tool designed to render models from FDF files. It allows users to explore 3D objects through different projections, apply rotations, zoom in and out, and toggle a menu for control. The program provides an interactive environment for viewing wireframe models and supports various projection techniques.

## Features

- 3D wireframe visualization
- Orthographic, isometric, and perspective projections
- Rotation and scaling transformations
- Zoom in and out
- User-friendly menu interface
- Keyboard controls for navigation

## Usage

1. Run the program using the following command:
 `python PyFDF.py <map.fdf>`


Replace `<map.fdf>` with the path to your FDF file.

2. Use the following keyboard controls to interact with the visualization:
- Press `R` and `T` to rotate around the X-axis.
- Press `F` and `G` to rotate around the Y-axis.
- Press `V` and `B` to rotate around the Z-axis.
- Press `K` to zoom in and `L` to zoom out.
- Press `Z` to scale the Z-axis up and `X` to scale it down.
- Press `Q` to toggle the menu on and off.
- Press `O` for orthographic projection, `I` for isometric projection, and `P` for perspective projection.

## Dependencies

- Python 3.x
- PyQt5
- numpy

## Installation

1. Clone this repository using:
    `git clone https://github.com/mimarque/PyFDF`

2. Navigate to the cloned directory:
    `cd PyFDF`

3. Install the required dependencies using pip:
    `pip install PyQt5`
    `pip install numpy`

4. Run the program as described in the [Usage](#usage) section.

## Mathematics and Implementation

PyFDF utilizes mathematical concepts to create its 3D wireframe visualizations from FDF files. This section provides an overview of the key mathematical concepts and their implementation in the program.

### 1. Coordinate Transformation

Wireframe models are defined using a collection of vertices (points in 3D space) and edges (line segments connecting vertices). PyFDF reads the FDF file to extract these vertices and edges.

For each vertex, a transformation matrix is applied to project the 3D coordinates onto a 2D plane for rendering. The transformation involves translation, rotation, and scaling to achieve the desired projection (orthographic, isometric, or perspective).

### 2. Projections

#### Orthographic Projection

Orthographic projection involves projecting each vertex onto the 2D plane without considering the distance from the viewer. This projection results in no foreshortening or depth perception, making it suitable for technical drawings.

#### Isometric Projection

Isometric projection is a special case of orthographic projection where the 3D object is rotated to align its edges with the axes of the 2D plane. This results in equal angles between all axes and provides a sense of depth without distortion.

#### Perspective Projection

Perspective projection simulates how objects appear smaller as they move farther away from the viewer. It involves applying a perspective transformation to each vertex based on its distance from the viewer. This projection is commonly used in realistic 3D rendering.

### 3. Rotation and Scaling

PyFDF allows users to interactively rotate the wireframe model around its axes. Rotation is achieved by applying rotation matrices to the vertices. Scaling is also possible along the X, Y, and Z axes, enabling users to control the size of the wireframe.

### 4. Rendering and User Interface

PyFDF uses the PyQt5 library to create a graphical user interface for interacting with the wireframe model. The wireframe is rendered using a graphics canvas, and the user can manipulate the view using keyboard controls. These controls trigger the transformation matrices and projection calculations, updating the display accordingly.
