import numpy as np

def create_spherical_spiral(fname, nvertices, afac=1.0, mint=-10.0, maxt=10.0):
    """Create a spherical spiral as a VTK polyline. The result is written
       into a VTK file in legacy format.
       Spherical spirals cut meridians at constant angle.
       They are given by the expression:
           
       x(t) = cos(t)*sin(c+pi/2)
       y(t) = sin(t)*sin(c+pi/2)
       z(t) = cos(c+pi/2)
           
       with c = arctan(a*t)
           
       where t is the curve parameter and a is the angle parameter. This
       vector field can be used to test VTK stream tracing.
    """

    # Curve parameter
    t = np.linspace(mint, maxt, nvertices)

    # Compute xyz coordinates of vertices
    vertex_coords = []
    for vertex in range(0, nvertices):
        c = np.arctan(afac*t[vertex])
        x = np.cos(t[vertex])*np.sin(c+0.5*np.pi)
        y = np.sin(t[vertex])*np.sin(c+0.5*np.pi)
        z = np.cos(c+0.5*np.pi)
        vertex_coords.append([x,y,z])

    nlines = nvertices - 1

    # Write into VTK file in legacy format
    vtk_file = open(fname, "w")
        
    vtk_file.write("# vtk DataFile Version 3.0\n")
    vtk_file.write("spherical spiral\n")
    vtk_file.write("ASCII\n")
    vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

    vtk_file.write("POINTS {} float\n".format(nvertices))
    for vertex in range(0, nvertices):
        vtk_file.write("{} {} {}\n".format(vertex_coords[vertex][0], vertex_coords[vertex][1], vertex_coords[vertex][2]))

    vtk_file.write("CELLS {} {}\n".format(nlines, nlines*3))
    for line in range(0, nlines):
        vtk_file.write("2 {} {}\n".format(line, line+1))

    vtk_file.write("CELL_TYPES {}\n".format(nlines))
    for cell in range(0, nlines):
        vtk_file.write("3\n") # VTK line type

    vtk_file.close

if __name__ == '__main__':
    create_spherical_spiral("spherical_spiral.vtk", 1000, 0.1, -20.0, 20.0)
