import numpy as np

class spherical_grid:
    """Provides methods for creating a spherical lat-lon grid using
       quadrilateral cells. The vertices and their connectivity are
       provided in a form that can easily be written into a VTK
       file.
       Also computes a vector field that with spherical spirals
       as streamlines, to test VTK streamline tracing.
       
       The code is written using plain Python for clarity.
    """
    def __init__(self, nloncells, nlatcells, radius, mintheta, maxtheta):

        self.nloncells = nloncells
        self.nlatcells = nlatcells

        # Grid is periodic in latitudinal (East-West) direction,
        # and bounded in longitudinal (North-South) direction
        self.nvertices = (nloncells+1)*nlatcells
        self.nlatedges = (nloncells+1)*nlatcells
        self.nlonedges =  nloncells   *nlatcells

        self.nedges = self.nlatedges+self.nlonedges
        self.ncells = nloncells*nlatcells

        self.radius = radius

        # Need to cut out the poles by restricting polar angle
        self.mintheta = mintheta
        self.maxtheta = maxtheta

        # Compute xyz coordinates and spherical angles for vertices
        self.set_vertex_coords_angles()

        # Compute vertex-cell and vertex-edge relations
        self.set_vertex_cell_connectivity()
        self.set_vertex_edge_connectivity()

    def set_vertex_coords_angles(self):
        """Creates a list of vertices with xyz and spherical coordinates.
        """
        self.vertex_coords = []
        self.vertex_angles = []
        # Grid is periodic in East-West direction
        for j in range(0,self.nloncells+1):
            for i in range(0,self.nlatcells):
                phi = i*2.0*np.pi/self.nlatcells
                theta = j*(self.maxtheta-self.mintheta)/self.nloncells + self.mintheta
                self.vertex_angles.append([phi, theta])
                self.vertex_coords.append([self.radius*np.cos(phi)*np.sin(theta),
                                           self.radius*np.sin(phi)*np.sin(theta),
                                           self.radius*np.cos(theta)])

        assert(len(self.vertex_angles) == self.nvertices)
        assert(len(self.vertex_coords) == self.nvertices)
        assert(self.vertex_angles[0] == [0., self.mintheta])

    def set_vertex_cell_connectivity(self):
        """Creates a list of quadrilateral cells, defined by vertex ID.
        """
        self.vert_on_cell = []
        # Grid is periodic in East-West direction
        for j in range(0,self.nloncells):
            for i in range(0,self.nlatcells):
                self.vert_on_cell.append([i+j*self.nlatcells,
                                          (i+1)%self.nlatcells+j*self.nlatcells,
                                          (i+1)%self.nlatcells+(j+1)*self.nlatcells,
                                          i+(j+1)*self.nlatcells])
        
        assert(len(self.vert_on_cell) == self.ncells)
        assert(self.vert_on_cell[0][0] == 0)
        assert(self.vert_on_cell[self.ncells-1][3] == (self.nvertices-1))

    def set_vertex_edge_connectivity(self):
        """Creates lists of cell edges, defined by vertex ID. The edges
           align with latitudes or longitudes, respectively.
        """

        # East-West
        self.vert_on_lat_edge = []
         # Need to include northermost edges
        for j in range(0,self.nloncells+1):
            for i in range(0,self.nlatcells):
                self.vert_on_lat_edge.append([i+j*self.nlatcells,
                                              (i+1)%self.nlatcells+j*self.nlatcells])

        assert(len(self.vert_on_lat_edge) == self.nlatedges)
        assert(self.vert_on_lat_edge[0][0] == 0)
        assert(self.vert_on_lat_edge[self.nlatedges-1][0] == (self.nvertices-1))

        # North-South
        self.vert_on_lon_edge = []
        for j in range(0,self.nloncells):
            for i in range(0,self.nlatcells):
                self.vert_on_lon_edge.append([i+j*self.nlatcells,
                                              i+(j+1)*self.nlatcells])
                                              
        assert(len(self.vert_on_lon_edge) == self.nlonedges)
        assert(self.vert_on_lon_edge[0][0] == 0)
        assert(self.vert_on_lon_edge[self.nlonedges-1][1] == (self.nvertices-1))

    def create_spiral_vector_field(self, afac=1.0):
        """Creates a vector field on the sphere with spherical spirals as
           streamlines. Spherical spirals cut meridians at constant angle.
           The parametric curve X(t) is given by the expressions:
           
           x(t) = cos(t)*sin(c+pi/2)
           y(t) = sin(t)*sin(c+pi/2)
           z(t) = cos(c+pi/2)
           
           with c = arctan(a*t)
           
           where t is the curve parameter and a is the angle parameter. The
           vector field V(theta, phi) is then given by
           
           V = dX/dt = Vphi + a/(1+a^2t^2) * Vtheta,
           
           where Vphi and Vtheta are the coordinate vector fields:
           
           Vtheta =  cos(theta)cos(phi)Ex + cos(theta)sin(phi)Ey - sin(theta)Ez
           Vphi   = -sin(theta)sin(phi)Ex + sin(theta)cos(phi)Ey

        """

        # Spherical spiral vector fields
        # Spiral parameter: afac = 0 => follow equator, afac >> 1 => follow meridian

        # Nodal
        self.point_vector = []
        for vertex in range(0, self.nvertices):
            # Compute angle factor for linear combination of coordinate vector fields
            theta = self.vertex_angles[vertex][1]
            t = np.tan(theta-0.5*np.pi)
            vangle = afac/(1.0+t*t)
            # We need these for computing the coordinate vector fields
            cosphi = np.cos(self.vertex_angles[vertex][0])
            sinphi = np.sin(self.vertex_angles[vertex][0])
            sintheta = np.sin(self.vertex_angles[vertex][1])
            costheta = np.cos(self.vertex_angles[vertex][1])
            # Coordinate vector fields in azimuthal (lat) and polar (lon) directions
            vlat = [-sintheta*sinphi, sintheta*cosphi, 0.0]
            vlon = [ costheta*cosphi, costheta*sinphi, -sintheta]
            # Linear combination to reproduce spherical spiral as a streamline
            # This is independent of phi - any starting point on the equator
            # shall produce a similar spherical spiral
            v = [vlat[0] + vlon[0]*vangle,
                 vlat[1] + vlon[1]*vangle,
                 vlat[2] + vlon[2]*vangle
                ]
            self.point_vector.append(v)

        # Cell-centered
        self.cell_vector = []
        for cell in range(0, self.ncells):
            # Compute spherical coordinates of cell centers
            phimean = 0.0
            thetamean = 0.0
            for vertex in range(0,4):
                phi = self.vertex_angles[self.vert_on_cell[cell][vertex]][0]
                if ( ((vertex == 1) | (vertex == 2)) & (phi == 0.0) ):
                    phi += 2.0*np.pi
                phimean += 0.25*phi
                thetamean += 0.25*self.vertex_angles[self.vert_on_cell[cell][vertex]][1]
            t = np.tan(thetamean-0.5*np.pi)
            vangle = afac/(1.0+t*t)
            cosphi = np.cos(phimean)
            sinphi = np.sin(phimean)
            sintheta = np.sin(thetamean)
            costheta = np.cos(thetamean)
            vlat = [-sintheta*sinphi, sintheta*cosphi, 0.0]
            vlon = [ costheta*cosphi, costheta*sinphi, -sintheta]
            v = [vlat[0] + vlon[0]*vangle,
                 vlat[1] + vlon[1]*vangle,
                 vlat[2] + vlon[2]*vangle
                ]
            self.cell_vector.append(v)

    def write_vtk_cells(self, fname):
        """Write mesh and vector fields in VTK legacy format. Vector
           fields are nodal and cell-centered.
        """

        vtk_file = open(fname, "w")
        
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Spherical quad grid\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        vtk_file.write("POINTS {} float\n".format(self.nvertices))
        for vertex in range(0, self.nvertices):
            vtk_file.write("{} {} {}\n".format(self.vertex_coords[vertex][0], self.vertex_coords[vertex][1], self.vertex_coords[vertex][2]))

        vtk_file.write("CELLS {} {}\n".format(self.ncells, self.ncells*5))
        for cell in range(0, self.ncells):
            vtk_file.write("4 {} {} {} {}\n".format(self.vert_on_cell[cell][0], self.vert_on_cell[cell][1], self.vert_on_cell[cell][2], self.vert_on_cell[cell][3]))

        vtk_file.write("CELL_TYPES {}\n".format(self.ncells))
        for cell in range(0, self.ncells):
            vtk_file.write("9\n") # VTK quad type

        vtk_file.write("POINT_DATA {}\n".format(self.nvertices))
        vtk_file.write("VECTORS point_vectorfield float\n")
        for vertex in range(0, self.nvertices):
            vtk_file.write("{} {} {}\n".format(self.point_vector[vertex][0],
                                               self.point_vector[vertex][1],
                                               self.point_vector[vertex][2]))

        vtk_file.write("CELL_DATA {}\n".format(self.ncells))
        vtk_file.write("VECTORS cell_vectorfield float\n")
        for cell in range(0, self.ncells):
            vtk_file.write("{} {} {}\n".format(self.cell_vector[cell][0],
                                               self.cell_vector[cell][1],
                                               self.cell_vector[cell][2]))

        vtk_file.close

    def write_vtk_edges(self, fname):
        """Write mesh in VTK legacy format. The mesh consists of cell edges.
        """

        vtk_file = open(fname, "w")
        
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Spherical edge grid\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        vtk_file.write("POINTS {} float\n".format(self.nvertices))
        for vertex in range(0, self.nvertices):
            vtk_file.write("{} {} {}\n".format(self.vertex_coords[vertex][0], self.vertex_coords[vertex][1], self.vertex_coords[vertex][2]))

        vtk_file.write("CELLS {} {}\n".format(self.nedges, self.nedges*3))
        for edge in range(0, self.nlonedges):
            vtk_file.write("2 {} {}\n".format(self.vert_on_lon_edge[edge][0], self.vert_on_lon_edge[edge][1]))
        for edge in range(0, self.nlatedges):
            vtk_file.write("2 {} {}\n".format(self.vert_on_lat_edge[edge][0], self.vert_on_lat_edge[edge][1]))

        vtk_file.write("CELL_TYPES {}\n".format(self.nedges))
        for cell in range(0, self.nedges):
            vtk_file.write("3\n") # VTK line type

        vtk_file.close

        
def create_spherical_grid():
    """Create a sphere using a lat-lon grid.
       The grid is periodic in East-West direction,
       and bounded in North-South direction to cut out
       the poles.
       The program also computes a nodal and cell-based
       vector fields with spherical spirals as streamlines.
    """
    nloncells = 10
    nlatcells = 10
    radius = 1.0
    # Set range of polar angles to define grid boundary in
    # North-South direction
    mintheta = 0.01*np.pi
    maxtheta = 0.99*np.pi
    grid = spherical_grid(nloncells, nlatcells, radius, mintheta, maxtheta)
    # Create spherical spiral vector field
    grid.create_spiral_vector_field(afac=0.1)
    grid.write_vtk_cells("spherical_cells.vtk")
    grid.write_vtk_edges("spherical_edges.vtk")

if __name__ == '__main__':
    create_spherical_grid()
