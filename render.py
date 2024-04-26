# Author: Joelene Hales, 2024

from OpenGL.GL import *
import glfw
import glm
import os
import sys
import numpy as np


# Initialize glfw
glfw.init() 
glfw.window_hint(glfw.SAMPLES, 4)  # Enable 4x multisampling

# Set context version required for shaders and VAOs
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # For MacOS
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

# Open a window and create its OpenGL context
screen_width = 800
screen_height = 800

window = glfw.create_window(screen_width, screen_height, "Marching Cubes", None, None)
glfw.make_context_current(window)

# Define filepath to project directory
project_directory = os.path.dirname(os.path.abspath(__file__))
shader_directory = os.path.join(project_directory, "Shaders")

# Helper functions
def readPLY(filename):
    """ Reads a PLY file.

    Parameters
    ----------
    filename : str
        Filepath to a PLY file to read.

    Returns
    -------
    vertices, normals, faces : list[float], list[float], list[int]
        Vertex positions, normals, and triangle faces in the triangle mesh.

    """

    # Get filepath to file
    directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(directory, filename)

    with open(filepath, "r") as file:

        # Read formatting and comments
        filetype = file.readline().strip("\n")
        format = file.readline().strip("\n")
        comment = file.readline().strip("\n")

        # Read number of vertices
        line = file.readline().strip().split(" ")
        num_vertices = int(line[2])

        # Create empty arrays to store each vertex's attributes
        vertex_data = {"x" : np.empty(num_vertices),
                        "y" : np.empty(num_vertices),
                        "z" : np.empty(num_vertices),
                        "nx" : np.empty(num_vertices),
                        "ny" : np.empty(num_vertices),
                        "nz" : np.empty(num_vertices),
                        "r" : np.empty(num_vertices),
                        "g" : np.empty(num_vertices),
                        "b" : np.empty(num_vertices),
                        "u" : np.empty(num_vertices),
                        "v" : np.empty(num_vertices)}

        # Determine which value in each vertex data line corresponds to each attribute
        attribute_indices = {}  # Stores each index in the line and the attribute it corresponds to
        line_num = 0            # Used to iterate until the end of the vertex property information
        while (line_num < len(vertex_data)): 

            line = file.readline().strip().split()  # Split the line's data into components

            if (line[0] == "element"):  # Beginning of triangle face properties begin
                break
            else:
                # Store the current position and its corresponding attribute
                attribute_indices[line_num] = line[2]
                line_num += 1

        # Read number of faces
        num_faces = int(line[2])

        # Skip format of triangle faces and end of header lines
        next(file)
        next(file)

        # Read vertex data from file
        for index in range(num_vertices):  # Iterate over each vertex

            vertex = file.readline().strip().split(" ")  # Split the line's data into components

            for i in range(len(vertex)):  # Iterate over each attribute

                attribute = attribute_indices[i]  # Lookup which attribute the value in this position corrresponds to
                vertex_data[attribute][index] = float(vertex[i])  # Add attribute to correct list

        # Read vertex positions and normals
        vertices = []
        normals = []

        for i in range(num_vertices):

            vertices.append(vertex_data['x'][i])
            vertices.append(vertex_data['y'][i])
            vertices.append(vertex_data['z'][i])

            normals.append(vertex_data['nx'][i])
            normals.append(vertex_data['ny'][i])
            normals.append(vertex_data['nz'][i])

        # Read triangle face indices from file
        faces = []
        while(len(faces) < num_faces):  # Repeat until all faces have been read

            # Read indices from file
            indices = file.readline().strip().split(" ")  # Split line into 3 indices
            faces.append(indices[1])
            faces.append(indices[2])
            faces.append(indices[3])

        # Convert to arrays
        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)

        return vertices, normals, faces


def matrix_to_array(matrix, dimensions):
    """ Converts a square matrix to an array. 
    
    Parameters
    ----------
    matrix : glm.mat4
        Matrix to convert to an array
    dimensions : int
        Number of rows and columns in the square matrix.

    Returns
    -------
    matrix_array : np.ndarray
        Matrix converted to an array.
    
    """
    
    matrix_array = np.array([matrix[i][j] for i in range(dimensions) for j in range(dimensions)])

    return matrix_array


# Main rendering classes
class Volume():
    """ Class used to draw a box around the marching volume. """

    def __init__(self, volume_min, volume_max):
        """ Initializes the shaders and vertex attributes used to render the
        marching volume.

        Parameters
        ----------
        volume_min : int
            Minimum value of x, y, and z in the marching volume.
        volume_max : int
            Maximum value of x, y, and z in the marching volume.
        
        """

        # Define corner vertices of marching volume
        volume_vertices = [volume_min, volume_min, volume_min,
                           volume_max, volume_min, volume_min,
                           volume_min, volume_min, volume_max,
                           volume_max, volume_min, volume_max,
                           volume_min, volume_max, volume_min,
                           volume_max, volume_max, volume_min,
                           volume_min, volume_max, volume_max,
                           volume_max, volume_max, volume_max]

        # Define vertex indices forming each edge of the marching volume
        endpoint_indices = [0, 1,
                            0, 2,
                            1, 3,
                            2, 3,
                            4, 5,
                            4, 6,
                            5, 7,
                            6, 7,
                            1, 5,
                            3, 7,
                            0, 4,
                            2, 6]

        # Convert to arrays
        volume_vertices = np.array(volume_vertices, dtype=np.float32)
        endpoint_indices = np.array(endpoint_indices, dtype=np.uint32)

        # Import shader codes from file
        vertex_shader_code = open(os.path.join(shader_directory, "simple.vs"), "r").read()
        fragment_shader_code = open(os.path.join(shader_directory, "simple.fs"), "r").read()

        # Create vertex and fragment shaders
        vertex_shader_ID = glCreateShader(GL_VERTEX_SHADER)
        fragment_shader_ID = glCreateShader(GL_FRAGMENT_SHADER)

        # Compile Vertex Shader
        glShaderSource(vertex_shader_ID, vertex_shader_code)
        glCompileShader(vertex_shader_ID)

        # Check for compilation error
        if not(glGetShaderiv(vertex_shader_ID, GL_COMPILE_STATUS)):
            raise RuntimeError(glGetShaderInfoLog(vertex_shader_ID))

        # Compile Fragment Shader
        glShaderSource(fragment_shader_ID, fragment_shader_code)
        glCompileShader(fragment_shader_ID)

        # Check for compilation error
        if not(glGetShaderiv(fragment_shader_ID, GL_COMPILE_STATUS)):
            raise RuntimeError(glGetShaderInfoLog(fragment_shader_ID))

        # Link shader program and attach shaders
        self.program_ID = glCreateProgram()

        glAttachShader(self.program_ID, vertex_shader_ID)
        glAttachShader(self.program_ID, fragment_shader_ID)
        glLinkProgram(self.program_ID)

        glDetachShader(self.program_ID, vertex_shader_ID)
        glDetachShader(self.program_ID, fragment_shader_ID)

        glDeleteShader(vertex_shader_ID)
        glDeleteShader(fragment_shader_ID)

        # Get handle for uniform variables
        self.MVP = glGetUniformLocation(self.program_ID, "MVP")
        self.color_uniform = glGetUniformLocation(self.program_ID, "color")

        # Create and bind VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # Create and bind VBO for vertex data
        self.vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, np.dtype(np.float32).itemsize*len(volume_vertices), volume_vertices, GL_STATIC_DRAW)

        # Set vertex attributes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,            # Attribute number
            3,            # Size (Number of components)
            GL_FLOAT,     # Type
            GL_FALSE,     # Normalized?
            0,            # Stride (Byte offset)
            None          # Offset
        )

        # Create and bind EBO for endpoint indices
        self.face_EBO = glGenBuffers(1)
        self.num_indices = len(endpoint_indices)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.face_EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.dtype(np.int32).itemsize*self.num_indices, endpoint_indices, GL_STATIC_DRAW)

        glBindVertexArray(0)  # Unbind VAO


    def draw(self, MVP):
        """ Renders a box around the marching volume. 
        
        Parameters
		----------
		MVP : glm.mat4
			Model view projection matrix.
            
        """

        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Use program
        glUseProgram(self.program_ID)

        # Set uniform variables
        glUniform3f(self.color_uniform, 1, 1, 1)  # White color
        glUniformMatrix4fv(self.MVP, 1, GL_FALSE, matrix_to_array(MVP, 4))

        # Bind VAO to restore captured state (buffer bindings and attribute specifications)
        glBindVertexArray(self.VAO)

        # Draw lines
        glDrawElements(GL_LINES, self.num_indices, GL_UNSIGNED_INT, None)

        # Unbind VAO and texture, clean up shader program
        glBindVertexArray(0)
        glUseProgram(0)

        glDisable(GL_BLEND)


class Axis():
    """ Class used to draw an axis of the marching voliume. """

    def __init__(self, which_axis, volume_min, volume_max):
        """ Initializes the shaders and vertex attributes used to render the
        marching volume.

        Parameters
        ----------
        which_axis : 'x', 'y', or 'z'
            Which axis to draw
        volume_min : int
            Minimum value of x, y, and z in the marching volume
        volume_max : int
            Maximum value of x, y, and z in the marching volume
        
        """

        # Define vertices and color depending on axis specified
        if (which_axis == 'x'):
            vertices = [volume_min, volume_min, volume_min,
                        volume_max, volume_min, volume_min]
            self.color = glm.vec3(1.0, 0.0, 0.0)
            

        elif (which_axis == 'y'):
            vertices = [volume_min, volume_min, volume_min,
                        volume_min, volume_max, volume_min]
            self.color = glm.vec3(0.0, 1.0, 0.0)

        else:
            vertices = [volume_min, volume_min, volume_min,
                        volume_min, volume_min, volume_max]
            self.color = glm.vec3(0.0, 0.0, 1.0)

        vertices = np.array(vertices, dtype=np.float32)
        self.num_vertices = len(vertices)

        # Import shader codes from file
        vertex_shader_code = open(os.path.join(shader_directory, "simple.vs"), "r").read()
        fragment_shader_code = open(os.path.join(shader_directory, "simple.fs"), "r").read()


        # Create vertex and fragment shaders
        vertex_shader_ID = glCreateShader(GL_VERTEX_SHADER)
        fragment_shader_ID = glCreateShader(GL_FRAGMENT_SHADER)

        # Compile Vertex Shader
        glShaderSource(vertex_shader_ID, vertex_shader_code)
        glCompileShader(vertex_shader_ID)

        # Check for compilation error
        if not(glGetShaderiv(vertex_shader_ID, GL_COMPILE_STATUS)):
            raise RuntimeError(glGetShaderInfoLog(vertex_shader_ID))

        # Compile Fragment Shader
        glShaderSource(fragment_shader_ID, fragment_shader_code)
        glCompileShader(fragment_shader_ID)

        # Check for compilation error
        if not(glGetShaderiv(fragment_shader_ID, GL_COMPILE_STATUS)):
            raise RuntimeError(glGetShaderInfoLog(fragment_shader_ID))

        # Link shader program and attach shaders
        self.program_ID = glCreateProgram()

        glAttachShader(self.program_ID, vertex_shader_ID)
        glAttachShader(self.program_ID, fragment_shader_ID)
        glLinkProgram(self.program_ID)

        glDetachShader(self.program_ID, vertex_shader_ID)
        glDetachShader(self.program_ID, fragment_shader_ID)

        glDeleteShader(vertex_shader_ID)
        glDeleteShader(fragment_shader_ID)

        # Get handle for each uniform
        self.MVP = glGetUniformLocation(self.program_ID, "MVP")
        self.color_uniform = glGetUniformLocation(self.program_ID, "color")

		# Create and bind VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

		# Create and bind VBO for vertex data
        self.vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, np.dtype(np.float32).itemsize*self.num_vertices, vertices, GL_STATIC_DRAW)

		# Set vertex attributes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,            # Attribute number
            3,            # Size (Number of components)
            GL_FLOAT,     # Type
            GL_FALSE,     # Normalized?
            0,            # Stride (Byte offset)
            None          # Offset
        )

        glBindVertexArray(0) # Unbind VAO


    def draw(self, MVP):
        """ Renders the axis. 
        
        Parameters
		----------
		MVP : glm.mat4
			Model view projection matrix.
            
        """

        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Use program
        glUseProgram(self.program_ID)

        # Set uniform variables
        glUniformMatrix4fv(self.MVP, 1, GL_FALSE, matrix_to_array(MVP, 4))
        glUniform3f(self.color_uniform, self.color.x, self.color.y, self.color.z)

        # Bind VAO to restore captured state (buffer bindings and attribute specifications)
        glBindVertexArray(self.VAO)

        # Draw axis
        glDrawArrays(GL_LINES, 0, self.num_vertices)

        # Unbind VAO and texture, clean up shader program
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)


class TriangleMesh():
    """ Class representing a triangle mesh generated by the marching cubes algorithm. """

    def __init__(self, filename):
        """ Reads the triangle mesh's attributes from a PLY file and initializes
        the shaders and vertex attributes used to render the object.
        
        Parameters
        ----------
        filename : str
            Filepath to the PLY file containing the triangle mesh's attributes.
    
        """

        # Read triangle mesh from PLY file
        vertices, normals, _ = readPLY(filename)  # Face indices not required when using glDrawArrays()
        self.num_vertices = len(vertices)


        # Import shader codes from file
        Phong_vertex_shader = open(os.path.join(shader_directory, "phong.vs"), "r").read()
        Phong_fragment_shader = open(os.path.join(shader_directory, "phong.fs"), "r").read()
        

        # Create vertex and fragment shaders
        vertex_shader_ID = glCreateShader(GL_VERTEX_SHADER)
        fragment_shader_ID = glCreateShader(GL_FRAGMENT_SHADER)

        # Compile Vertex Shader
        glShaderSource(vertex_shader_ID, Phong_vertex_shader)
        glCompileShader(vertex_shader_ID)

        # Check for compilation error
        if not(glGetShaderiv(vertex_shader_ID, GL_COMPILE_STATUS)):
            raise RuntimeError(glGetShaderInfoLog(vertex_shader_ID))

        # Compile Fragment Shader
        glShaderSource(fragment_shader_ID, Phong_fragment_shader)
        glCompileShader(fragment_shader_ID)

        # Check for compilation error
        if not(glGetShaderiv(fragment_shader_ID, GL_COMPILE_STATUS)):
            raise RuntimeError(glGetShaderInfoLog(fragment_shader_ID))

        # Link shader program and attach shaders
        self.program_ID = glCreateProgram()

        glAttachShader(self.program_ID, vertex_shader_ID)
        glAttachShader(self.program_ID, fragment_shader_ID)
        glLinkProgram(self.program_ID)

        glDetachShader(self.program_ID, vertex_shader_ID)
        glDetachShader(self.program_ID, fragment_shader_ID)

        glDeleteShader(vertex_shader_ID)
        glDeleteShader(fragment_shader_ID)

        # Get handle for each uniform
        self.MVP = glGetUniformLocation(self.program_ID, "MVP")
        self.V = glGetUniformLocation(self.program_ID, "V")
        self.M = glGetUniformLocation(self.program_ID, "M")
        self.LightDir = glGetUniformLocation(self.program_ID, "LightDir")
        self.modelColor = glGetUniformLocation(self.program_ID, "modelColor")

		# Create and bind VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

		# Create and bind VBO for vertex data
        self.vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, np.dtype(np.float32).itemsize*self.num_vertices, vertices, GL_STATIC_DRAW)

		# Set vertex attributes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,            # Attribute number
            3,            # Size (Number of components)
            GL_FLOAT,     # Type
            GL_FALSE,     # Normalized?
            0,            # Stride (Byte offset)
            None          # Offset
        )

		# Create and bind VBO for normal vectors
        self.normal_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_VBO)
        glBufferData(GL_ARRAY_BUFFER, np.dtype(np.float32).itemsize*self.num_vertices, normals, GL_STATIC_DRAW)

		# Set normal attributes
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,            # Attribute number
            3,            # Size (Number of components)
            GL_FLOAT,     # Type
            GL_TRUE,      # Normalized?
            0,            # Stride (Byte offset)
            None          # Offset
        )

        glBindVertexArray(0)  # Unbind VAO


    def draw(self, MVP, V, M, LightDir, modelColor):
        """ Renders the triangle mesh. 
        
        Parameters
		----------
		MVP : glm.mat4
			Model view projection matrix.
        V : glm.mat4
            View matrix
        M : glm.mat4
            Model matrix
        LightDir : glm.vec3
            Direction of the light source
        modelColor : glm.vec3
            Object's base color
            
        """
        
        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		# Use program
        glUseProgram(self.program_ID)

        # Set uniform variables
        glUniformMatrix4fv(self.MVP, 1, GL_FALSE, matrix_to_array(MVP, 4))
        glUniformMatrix4fv(self.V, 1, GL_FALSE, matrix_to_array(V, 4))
        glUniformMatrix4fv(self.M, 1, GL_FALSE, matrix_to_array(M, 4))
        glUniform3f(self.LightDir, LightDir.x, LightDir.y, LightDir.z)
        glUniform4f(self.modelColor, modelColor.x, modelColor.y, modelColor.z, 1.0)

		# Bind VAO to restore captured state (buffer bindings and attribute specifications)
        glBindVertexArray(self.VAO)

        # Draw triangles
        glDrawArrays(GL_TRIANGLES, 0, self.num_vertices)

        # Unbind VAO and texture, clean up shader program
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)



class Camera():
    """ Stores the position of a camera which supports "globe" movements using spherical coordinates. """

    def __init__(self, center):
        """ Initializes the camera position.

        Parameters
        ----------
        center : glm.vec3
            Camera position, in Cartesian coordinates.
        
        """

        # Validate input vectors
        if (type(center) != glm.vec3):
            raise TypeError("Center position must be given as GLM 3D vectors.") 


        # Calculate radius, theta, and phi values by converting center coordinate from Cartesian spherical coordinates
        self.radius = np.sqrt(center.x**2 + center.y**2 + center.z**2)
        self.theta = np.arctan(center.z / center.x)
        self.phi = np.arccos(center.y / self.radius)


    def rotatePhi(self, increment):
        """ Increments the polar angle phi by a given amount, in radians. Has
        the effect of rotating the camera up or down. Clamps the angle between
        epsilon and pi - epsilon to prevent the camera from flipping.
        
        Parameters
        ----------
        increment : float
            Amount to increment the angle by, in radians.
        
        """

        self.phi += increment
        
        # Clamp between epsilon and pi - epsilon to prevent vector from flipping and producing unwanted effects
        epsilon = 0.001
        self.phi = max(min(self.phi, max(epsilon, np.pi - epsilon)), min(epsilon, np.pi - epsilon))


    def rotateTheta(self, increment):
        """ Increments the azimuthal angle theta by a given amount, in radians.
        Has the effect of rotating the camera to spin around at a fixed elevation.
        
        Parameters
        ----------
        increment : float
            Amount to increment the angle by, in radians.
        
        """

        self.theta += increment


    def zoomRadius(self, increment):
        """ Increments the radius by a given amount. Has the effect of zooming
        the camera in for a negative value, or out for a positive value. The
        camera is stopped before the radius becomes negative.
        
        Parameters
        ----------
        increment : float
            Amount to increment the radius.
        
        """

        self.radius += increment

        # Do not let radius become negative
        if (self.radius <= 0.0001):
            self.radius = 0.0001


    def get_position(self):
        """ Calculates the camera's position in Cartesian coordinates. 
        
        Returns
        -------
        position : glm.vec3
            Camera's position, in Cartesian coordinates.

        """

        # Calculate each coordinate from spherical coordinates
        x = self.radius * np.cos(self.theta) * np.sin(self.phi)
        y = self.radius * np.cos(self.phi)
        z = self.radius * np.sin(self.theta) * np.sin(self.phi)

        position = glm.vec3(x, y, z)  # Camera position, in Cartesian coordinates

        return position


# Validate input and unpack arguments
if (len(sys.argv) == 4):  # Color name not given
    _, filename, volume_min, volume_max = sys.argv
    color_name = "turquoise"
elif (len(sys.argv) == 5):  # Color name given
    _, filename, volume_min, volume_max, color_name = sys.argv
else:  # Invalid input
    raise TypeError("Expected 3 or 4 arguments but {} were given.".format(len(sys.argv) - 1))


# Define RGB value for given color name
if (color_name.lower() == "turquoise"):
    model_color = glm.vec3(0.0, 1.0, 1.0)
elif (color_name.lower() == "fuchsia"):
    model_color = glm.vec3(1.0, 0.0, 0.667)
elif (color_name.lower() == "lime"):
    model_color = glm.vec3(0.667, 1.0, 0.0)
elif (color_name.lower() == "orange"):
    model_color = glm.vec3(1.0, 0.667, 0.0)
elif (color_name.lower() == "purple"):
    model_color = glm.vec3(0.667, 0.0, 1.0)
else:
    raise ValueError("Undefined color: '{}'. Please enter one of the following options:\n'turquoise', 'lime', 'orange', 'purple', 'fuchsia'".format(color_name))


# Create triangle mesh
triangle_mesh = TriangleMesh(filename)

volume_min = int(volume_min)  # Type cast integer inputs
volume_max = int(volume_max)

# Create objects to render marching volume boundary and axes
volume = Volume(volume_min, volume_max)
x_axis = Axis('x', volume_min, volume_max)
y_axis = Axis('y', volume_min, volume_max)
z_axis = Axis('z', volume_min, volume_max)

# Define matrices
P = glm.perspective(glm.radians(75.0), screen_width/screen_height, 0.001, 1000.0)  # Projection matrix with a vertical field of view of 75 degrees
M = glm.mat4(1.0)  # Model matrix

# Initialize camera settings, in Cartesian coordinates
eye = glm.vec3(0.0, 0.0, 0.0)      # Camera look direction
up = glm.vec3(0.0, 1.0, 0.0)       # Direction of up
center = glm.vec3(5.0, 5.0, 5.0)   # Camera position

camera = Camera(center)

# Initialize variables used to move camera
x_press = None  # Location mouse was first clicked
y_press = None
dx = 0          # Distance mouse was dragged in each direction
dy = 0
dr = 0          # Proportional to (time up arrow key was held) - (time down arrow key was held)

# Define direction of light source
light_direction = glm.vec3(0.0, 10.0, 10.0)

# Ensure we can capture the escape key being pressed
glfw.set_input_mode(window, glfw.STICKY_KEYS, GL_TRUE)

# Ensure depth is determined correctly
glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LESS)

# Render loop
while (glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window)):  # Repeat until escape key is pressed or window is closed

    # Clear buffers and set background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.2, 0.2, 0.3, 0.0)  # Dark blue

    glfw.poll_events()  # Poll for events

    # Process up and down arrow key presses
    if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):  # Up arrow is held
        dr -= 0.01  # Move inward radially
    elif (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):  # Down arrow is held
        dr += 0.01  # Move outward radially
    else:  # No arrow key is held
        dr = 0   # Stop moving

    # Process mouse click and drag
    mouse_left_state = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
    if (mouse_left_state == glfw.PRESS):  # Left mouse button is clicked

        # Get location clicked
        x_cursor, y_cursor = glfw.get_cursor_pos(window)

        if x_press is None:  # On the first frame mouse is clicked
            x_press = x_cursor  # Save location mouse was first clicked
            y_press = y_cursor

        # Calculate distance between initial location clicked and position dragged to
        dx = x_cursor - x_press
        dy = y_cursor - y_press

        # Save the dragged cursor positon to be used in calculation for next frame
        x_press = x_cursor
        y_press = y_cursor

    elif (mouse_left_state == glfw.RELEASE):  # Left mouse button is released
        x_press = None    # Reset clicked location
        y_press = None
        dx = 0            # Stop moving after release
        dy = 0


    # Adjust camera position based on click and drag motion
    camera.zoomRadius(dr)            # Modify radius based on arrow press
    camera.rotateTheta(0.005 * dx)   # Modify theta based on horizontal drag motion
    camera.rotatePhi(-0.005 * dy)    # Modify phi based on vertical drag motion

    # Set view matrix to use new camera position
    V = glm.lookAt(camera.get_position(), eye, up)

    # Calculate model view projection matrix
    MVP = P * V * M

    # Draw marching volume boundary and axes
    x_axis.draw(MVP)
    y_axis.draw(MVP)
    z_axis.draw(MVP)
    volume.draw(MVP)

    # Render triangle mesh
    triangle_mesh.draw(MVP, V, M, light_direction, model_color)

    # Swap buffers
    glfw.swap_buffers(window)
    glfw.poll_events()


# Close OpenGL window and terminate GLFW
glfw.terminate()
