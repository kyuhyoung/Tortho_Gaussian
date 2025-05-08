from plyfile import PlyData

plydata = PlyData.read('')

print("Properties in the PLY file:")
for property in plydata.elements[0].properties:
    print(property.name)
