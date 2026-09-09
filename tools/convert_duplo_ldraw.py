"""Convert the bundled attributed LDraw subset to centered, millimetre OBJ meshes."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / 'assets/duplo/ldraw'


def triangles(path, stack=()):
    if path in stack:
        raise ValueError(f'Cyclic LDraw reference: {path}')
    result = []
    ccw, invert = True, False
    for line in path.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == '0':
            if 'BFC' in t:
                if 'CCW' in t:
                    ccw = True
                elif 'CW' in t:
                    ccw = False
                if 'INVERTNEXT' in t:
                    invert = True
            continue
        if t[0] == '1':
            offset = np.array(t[2:5], dtype=float)
            matrix = np.array(t[5:14], dtype=float).reshape(3, 3)
            name = ' '.join(t[14:]).replace('\\', '/').lower()
            child = next((LIB / prefix / name for prefix in ('parts', 'p')
                          if (LIB / prefix / name).is_file()), None)
            if child is None:
                raise FileNotFoundError(name)
            vertices = triangles(child, stack + (path,)) @ matrix.T + offset
            if invert ^ (np.linalg.det(matrix) < 0):
                vertices = vertices[:, ::-1, :]
            result.extend(vertices)
            invert = False
        elif t[0] in ('3', '4'):
            points = np.array(t[2:], dtype=float).reshape(-1, 3)
            for indices in ((0, 1, 2),) if t[0] == '3' else ((0, 1, 2), (0, 2, 3)):
                face = points[list(indices)]
                result.append(face if ccw else face[::-1])
        elif t[0] not in ('2', '5'):
            raise ValueError(f'Unsupported LDraw line: {line}')
    return np.asarray(result, dtype=float).reshape(-1, 3, 3)


def main():
    metadata = {}
    for label, part in [('duplo_2x2', '3437'), ('duplo_2x4', '3011')]:
        faces = triangles(LIB / 'parts' / f'{part}.dat')
        # LDraw: 1 unit = 0.4 mm; +Y down. Object frame: X long, Z up through studs.
        rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        faces = faces @ rotation.T * .4
        low, high = faces.min(axis=(0, 1)), faces.max(axis=(0, 1))
        faces -= (low + high) / 2
        output = ROOT / 'assets/duplo' / f'{label}.obj'
        with output.open('w') as f:
            f.write('# Derived from LDraw; attribution/license in README.md and ldraw/CAreadme.txt\n')
            f.write('# Units: millimetres; origin: full mesh bounding-box center; Z up\n')
            for point in faces.reshape(-1, 3):
                f.write('v ' + ' '.join(f'{x:.8g}' for x in point) + '\n')
            for i in range(len(faces)):
                f.write(f'f {3*i+1} {3*i+2} {3*i+3}\n')
        metadata[label] = dict(part_id=part, mesh=output.name, units='mm',
                               dimensions_mm=(high-low).tolist(),
                               origin='full mesh bounding-box center including studs',
                               axes={'x':'long side (2x4)', 'y':'short side', 'z':'up through studs'},
                               triangles=len(faces))
        print(label, metadata[label])
    (ROOT / 'assets/duplo/meshes.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
