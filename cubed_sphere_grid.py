# Cubed-sphere mesh generation in this program is based on Alex Pletzer's "igCubedSphere.py"
# (https://github.com/pletzer/inugrid/blob/master/py/igCubedSphere.py)

import numpy as np
import vtk

class CubedSphere:

    def __init__(self, numCellsPerTile, radius=1.0):

        self.radius = radius

        # VTK object for stitching cube faces together
        self.appendGrids = vtk.vtkAppendFilter()
        self.appendGrids.MergePointsOn()
        self.appendGrids.SetOutputPointsPrecision(1) # double

        # Create tile grids
        numPointsPerTile = numCellsPerTile + 1
        us = np.linspace(0., 1., numPointsPerTile)
        vs = np.linspace(0., 1., numPointsPerTile)
        uu, vv = np.meshgrid(us, vs)
        # box is [0, 1] x [0, 1], let's fit a sphere inside the box
        centre = np.array([0.5, 0.5, 0.5])

        self.xyzList = []
        self.areaList = []
        self.tileXyzList = []
        self.tilePtsList = []
        self.tileGridList = []
        self.tileAreaList = []

        # Create grid by appending 6 tiles

        # iterate over the space dimensions
        for dim0 in range(3):

            # low or high side
            for pm in range(-1, 2, 2):

                # normal vector, pointing out
                normal = np.zeros((3,), np.float64)
                normal[dim0] = pm

                # indices of the dimensions on the tile
                dim1 = (dim0 + 1) % 3
                dim2 = (dim0 + 2) % 3

                # Vertex coordinates
                xyz = np.zeros((numPointsPerTile*numPointsPerTile, 3), 
                                np.float64)

                # grid on the box's side/tile 
                xyz[:, dim0] = (pm + 1.0)/2.
                xyz[:, dim1] = uu.flat
                xyz[:, dim2] = vv.flat
                # fix the vertex ordering so the area points outwards
                if pm > 0:
                    xyz[:, dim1] *= -1.0
                    xyz[:, dim1] += 1.0

                # Project the vertices onto sphere
                for i in range(3):
                    xyz[:, i] -= centre[i]
                dist = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
                for i in range(3):
                    # normalize
                    xyz[:, i] /= dist
                    # extend to the sphere's surface
                    xyz[:, i] *= radius

                ntot = numPointsPerTile**2

                # Create VTK structured grid for this tile
                tileXyz = vtk.vtkDoubleArray()
                tileXyz.SetNumberOfComponents(3)
                tileXyz.SetNumberOfTuples(ntot)
                tileXyz.SetVoidArray(xyz, 3*ntot, 1)

                tilePts = vtk.vtkPoints()
                tilePts.SetNumberOfPoints(ntot)
                tilePts.SetData(tileXyz)

                tileGrid = vtk.vtkStructuredGrid()
                tileGrid.SetDimensions(numPointsPerTile, numPointsPerTile, 1)
                tileGrid.SetPoints(tilePts)

                self.appendGrids.AddInputData(tileGrid)

                # Need to keep these alive to avoid dangling pointers
                self.xyzList.append(xyz)
                self.tileXyzList.append(tileXyz)
                self.tilePtsList.append(tilePts)
                self.tileGridList.append(tileGrid)

        # Stitch grids together
        self.appendGrids.Update()
        self.grid = self.appendGrids.GetOutput() # vtkUnstructuredGrid object

    def set_spiral_vector_field(self, afac=1.0):
                
        gridPoints = self.grid.GetPoints() # vtkPoints object
        npoints = gridPoints.GetNumberOfPoints()
        ncells = self.grid.GetNumberOfCells()

        # Create new point vector field
        spiral_vfield_p = vtk.vtkDoubleArray()
        spiral_vfield_p.SetName('spiral_vfield_points')
        spiral_vfield_p.SetNumberOfComponents(3) # Vector
        spiral_vfield_p.SetNumberOfTuples(npoints)

        # Compute point vector field
        for pointid in range(0, npoints):
            point = gridPoints.GetPoint(pointid)
            r, theta, phi = self.cart2spherical(point[0], point[1], point[2])
            vfield = self.compute_vfield(afac, theta, phi)
            spiral_vfield_p.SetComponent(pointid, 0, vfield[0])
            spiral_vfield_p.SetComponent(pointid, 1, vfield[1])
            spiral_vfield_p.SetComponent(pointid, 2, vfield[2])

        # Store point data
        pointData = self.grid.GetPointData() # vtkPointData object
        pointData.SetVectors(spiral_vfield_p)

        # Create new cell vector field
        spiral_vfield_c = vtk.vtkDoubleArray()
        spiral_vfield_c.SetName('spiral_vfield_cells')
        spiral_vfield_c.SetNumberOfComponents(3) # Vector
        spiral_vfield_c.SetNumberOfTuples(ncells)

        # Compute cell vector field
        for cellid in range(0, ncells):
            
            # Query point IDs for this cell
            pts = vtk.vtkIdList()
            self.grid.GetCellPoints(cellid, pts) # vtkCellArray
            nVertexIds = pts.GetNumberOfIds()
            
            phi = np.zeros(4, np.float64)
            theta = np.zeros(4, np.float64)
            
            # Compute spherical coordinates of each point
            for vertexid in range(0, nVertexIds):
                point = gridPoints.GetPoint(pts.GetId(vertexid))
                r, theta[vertexid], phi[vertexid] = self.cart2spherical(point[0], point[1], point[2])
            # Shift phi by 2pi for cells that touch or cross the meridian and average coordinates
            if ( (phi.max() - phi.min()) > np.pi ):
                phi[np.where(phi < 0.5*np.pi)] += 2.0*np.pi
            phimean = phi.mean()
            thetamean = theta.mean()
            vfield = self.compute_vfield(afac, thetamean, phimean)
            spiral_vfield_c.SetComponent(cellid, 0, vfield[0])
            spiral_vfield_c.SetComponent(cellid, 1, vfield[1])
            spiral_vfield_c.SetComponent(cellid, 2, vfield[2])

        # Store cell data
        cellData = self.grid.GetCellData() # vtkCellData object
        cellData.SetVectors(spiral_vfield_c)

    def cart2spherical(self, x, y, z):
        """Convert Cartesian coordinates to spherical coordinates
        """
        r = np.sqrt(x*x+y*y+z*z)
        rxy = np.sqrt(x*x+y*y)
        if (rxy != 0.0):
            phi = np.arccos(np.abs(x)/rxy)
        else:
            phi = 0.0
        if ( (x < 0) & (y >= 0) ):
            phi = np.pi - phi
        if ( (x < 0) & (y < 0) ):
            phi += np.pi
        if ( (x >= 0) & (y < 0) ):
            phi = 2.0*np.pi - phi
        theta = np.arccos(z/r)
        return r, theta, phi

    def compute_vfield(self, afac, theta, phi):
        """Computes the spiral curve vector field given parameter afac,
           polar angle theta, and azimuth angle phi.
        """
        # Compute spiral curve parameter given polar angle
        # and scale factor for coordiante vector field
        t = np.tan(theta-0.5*np.pi)
        vangle = afac/(1.0+t*t)
        # We need these for computing the coordinate vector fields
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        # Coordinate vector fields in azimuthal (lat) and polar (lon) directions
        vlat = np.array([-sintheta*sinphi, sintheta*cosphi, 0.0], np.float64)
        vlon = np.array([ costheta*cosphi, costheta*sinphi, -sintheta], np.float64)
        # Linear combination to reproduce spherical spiral as a streamline
        # This is independent of phi - any starting point on the equator
        # shall produce a similar spherical spiral
        vfield = vlat + vangle*vlon

        return vfield

    def write_vtk_file(self, filename):
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(self.grid)
        writer.Update()

    def show(self):

        gridMapper = vtk.vtkDataSetMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            gridMapper.SetInputConnection(self.grid.GetProducerPort())
        else:
            gridMapper.SetInputData(self.grid)
            gridMapper.Update()

        gridActor = vtk.vtkActor()
        gridActor.SetMapper(gridMapper)
        gridActor.GetProperty().SetColor(93./255., 173./255., 226./255.)

        ren = vtk.vtkRenderer()
        ren.AddActor(gridActor)
        ren.SetBackground(1, 1, 1)
        ren.ResetCamera()
        
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(640, 640)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.Initialize()
        iren.Start()

def create_cubed_sphere():
    numCells = 10
    cs = CubedSphere(numCells)
    cs.set_spiral_vector_field(afac=0.0)
    cs.write_vtk_file('cubed-sphere.vtk')
    cs.show()

if __name__ == '__main__':
    create_cubed_sphere()
