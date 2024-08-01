# Shortest_Flip_Sequences
Contains the code and data for "Finding Shortest Flip Sequences Between Connected Graph Partitions" paper.

## Data
The **GridShapefiles** folder contains several shapefiles for square grids. The **Grid_Plans** folder contains assignment files for several horizontal (A) and vertical (B) stripe plans. The **PaperExperiments** folder contains several randomly generated 2-, 4-, 6- and 8-partitions on 20x20, 12x12, 24x24, and 40x40 grid graphs, respectively, and 5- and 8-partitions on the complete graphs K<sub>100</sub> and K<sub>200</sub>, respectively (used for paper experiments).

## Code

The **Environments_GurobiGerrychain** folder contains environment files that detail the versions of all packages that the code uses.

The **astar.py** code contains functions to run A* search and the betapoint heuristic to find a shortest flip sequence between two connected partitions.

The **midpoint_paths.py** code contains functions to determine a sequence of fractional points (betapoints) between two given partitions.

The **Examples.ipynb** code contains example calls to the functions in astar.py using the data listed above.
