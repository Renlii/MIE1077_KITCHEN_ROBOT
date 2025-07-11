import trimesh
import sys
import json
from glob import glob
import xml.etree.ElementTree as ET
import os
# file = sys.argv[1]

def find_scale(xml):
    scale = [1, 1, 1]

    # read xml
    tree = ET.parse(xml) 
    root = tree.getroot()

    # iterate and find <scale>
    for mesh in root.iter('mesh'):
        scale_el = mesh.find('scale')
        if scale_el is not None:
            scale_text = scale_el.text.strip()
            scale_vals = list(map(float, scale_text.split()))
            scale = scale_vals

    return scale

def find_unit(xml):
    # read xml
    tree = ET.parse(xml) 
    root = tree.getroot()

    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}

    unit_el = root.find('.//c:unit', ns)
    if unit_el is not None:
        return float(unit_el.attrib.get('meter', '1.0'))
    return 1.0


files = glob("*/meshes/*.dae")
for file in files:
    # 读取你的 DAE 文件（支持 DAE, STL, OBJ 等）
    mesh = trimesh.load(file)

    # 获取 axis-aligned bounding box (AABB)
    bounding_box = mesh.bounding_box.bounds
    min_corner = bounding_box[0]
    max_corner = bounding_box[1]

    # 实际尺寸 = max - min
    size = max_corner - min_corner

    # find scale
    model_xml = os.path.dirname(file) + "/../model.sdf"
    scale = find_scale(model_xml)

    # find unit
    meter = find_unit(file)

    print(file, "scale", scale, "unit", meter)
    min_corner = list(min_corner)
    max_corner = list(max_corner)
    size = list(size)
    for i in range(len(scale)):  
        min_corner[i] *= scale[i] * meter
        max_corner[i] *= scale[i] * meter
        size[i] *= scale[i] * meter

    out = {"bounding_box_min": list(min_corner), "bounding_box_max": list(max_corner), "model_size": list(size)}

    outfile = file.replace(".dae", ".json")
    assert(outfile != file)
    json.dump(out, open(outfile, 'w'), indent=4)


# print(out)
# print(f"Bounding Box Min: {min_corner}")
# print(f"Bounding Box Max: {max_corner}")
# print(f"Model Size (x, y, z): {size}")
