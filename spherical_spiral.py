import numpy as np
import vtk

class spherical_spiral:
    """Creates a spherical spiral as a VTK polyline object.
       Also defines methods for visualising and storing the
       spiral.
    """

    def __init__(self, nvertices, afac=1.0, mint=-10.0, maxt=10.0):

        self.nvertices = nvertices
        self.ncells = nvertices - 1
        self.afac = afac
        self.mint = mint
        self.maxt = maxt

        self.create_spherical_spiral()

    def create_spherical_spiral(self):
        """Create a spherical spiral as a VTK polyline.

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
        t = np.linspace(self.mint, self.maxt, self.nvertices)

        # Compute and store xyz coordinates of vertices ("points")
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(self.nvertices)
        for vertex in range(0, self.nvertices):
            c = np.arctan(self.afac*t[vertex])
            x = np.cos(t[vertex])*np.sin(c+0.5*np.pi)
            y = np.sin(t[vertex])*np.sin(c+0.5*np.pi)
            z = np.cos(c+0.5*np.pi)
            points.SetPoint(vertex, x, y, z)

        # Set point connectivity to form line
        cells = vtk.vtkCellArray()
        for cell in range(0, self.ncells):
            cells.InsertNextCell(2)
            cells.InsertCellPoint(cell)
            cells.InsertCellPoint(cell+1)

        # Create VTK object
        self.polygon = vtk.vtkPolyData()
        self.polygon.SetPoints(points)
        self.polygon.SetLines(cells)

    def show(self):
        """Visualise spherical spiral
        """

        polygonMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polygonMapper.SetInputConnection(self.polygon.GetProducerPort())
        else:
            polygonMapper.SetInputData(self.polygon)
            polygonMapper.Update()

        polygonActor = vtk.vtkActor()
        polygonActor.SetMapper(polygonMapper)

        ren = vtk.vtkRenderer()
        ren.AddActor(polygonActor)
        ren.SetBackground(0.1, 0.2, 0.4)
        ren.ResetCamera()

        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(300, 300)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.Initialize()
        iren.Start()

    def write_vtk_file(self, fname):
        """Write poly line into a ASCII VTK file.
        """

        vtk_file = vtk.vtkPolyDataWriter()
        vtk_file.SetFileName(fname)
        vtk_file.SetInputData(self.polygon)
        vtk_file.Update()

if __name__ == '__main__':
    spiral = spherical_spiral(1000, 0.1, -20.0, 20.0)
    spiral.show()
    spiral.write_vtk_file("spherical_spiral.vtk")
