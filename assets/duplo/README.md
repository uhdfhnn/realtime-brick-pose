# DUPLO geometry for initial pose inspection

Derived from LDraw models 3437 (2x2) and 3011 (2x4), authored by Tony Hafner
[hafhead], with subsequent edits and primitive authors credited in each bundled
source file. These are community models, not LEGO manufacturing CAD.

Source: https://github.com/gkjohnson/ldraw-parts-library/tree/1f24cac1821b9a86a5a84cd3a40291aad24679b3/complete/ldraw
This is a historical mirror snapshot, not a claim of the latest LDraw geometry.
Original source headers and the supplied CAreadme.txt / CAlicense.txt are retained
in ldraw/. The subset declares CCAL 2.0; see those files for attribution and terms.
The OBJ files are converted adaptations of that attributed source geometry.

Rebuild with `python tools/convert_duplo_ldraw.py` (NumPy required). Conversion
recursively expands subparts, triangulates quads, handles BFC winding/inversions,
converts LDraw units to mm (0.4 mm/unit), rotates +Y-down to +Z-up, and centers the
full mesh bounding box, including studs. Lines/conditional edges are not surfaces
and are excluded. Source subparts are included for offline reproducibility.

Computed mesh bounds:
- 2x2: 32 x 32 x 23.6 mm; 572 triangles.
- 2x4: 64 x 32 x 23.6 mm; 1180 triangles.

These include studs. They are nominal CAD extents, not measured manufacturing
clearances. Compare to the actual bricks before trusting metric translations.
The origin is not the brick bottom or stud plane. Geometric symmetry remains
ambiguous (2x2 quarter turns, 2x4 half turns); snapshot inference does not promise
an identifiable stud-index orientation or symmetry-continuous tracking.
