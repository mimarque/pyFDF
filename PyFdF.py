from os import path
import sys
from math import sqrt, cos, sin, tan, radians
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap, QLinearGradient, QFont, QKeyEvent
from PyQt5.QtCore import Qt, QPoint, QEvent
import numpy as np
import time


class MyWarning(UserWarning):
    pass

class Vec4_np:
    __slots__ = ['xyzw', 'c']

    def __init__(self, x: float = 0, y: float = 0, z: float = 0, w: float = 0, color=[0, 0, 0, 1.0]):
        self.xyzw = np.array([x, y, z, w])
        self.c = color

    def copy(self):
        return Vec4_np(self.x, self.y, self.z, self.w, self.c)
    
    def get_xyzw(self):
        return self.xyzw
    
    def get_x(self):
        return self.xyzw[0]
    
    def get_y(self):
        return self.xyzw[1]
    
    def get_z(self):
        return self.xyzw[2]
    
    def get_w(self):
        return self.xyzw[3]

    def get_rgba(self):
        return self.c

    def set_xyzw(self, x: float = 0, y: float = 0, z: float = 0, w: float = 0):
        self.xyzw = np.array([x, y, z, w])

    def set_x(self, x: float = 0):
        self.xyzw[0] = x

    def set_y(self, y: float = 0):
        self.xyzw[1] = y

    def set_z(self, z: float = 0):
        self.xyzw[2] = z

    def set_w(self, w: float = 0):
        self.xyzw[3] = w

    def set_rgba(self, color=[0, 0, 0, 1.0]):
        self.c = color

    #define a length method to calculate the length of a vector
    def length(self):
        return np.linalg.norm(self.xyzw)

    def normalize(self):
        length = self.length()
        if length != 0:
            self.xyzw /= length
        return self

    def normalize_new(self):
        length = self.length()
        if length != 0:
            return Vec4_np(*(self.xyzw / length), self.c)
        else:
            return Vec4_np(self.xyzw, self.c)

    def dot(self, other: 'Vec4_np'):
        return np.dot(self.xyzw, other.xyzw)

    def cross(self, other: 'Vec4_np'):
        x, y, z, _ = self.xyzw
        x2, y2, z2, _ = other.xyzw
        xyz = np.array([x, y, z])
        xyz2 = np.array([x2, y2, z2])
        return Vec4_np(*(np.cross(xyz, xyz2)), 1, self.c)

    def __add__(self, other: 'Vec4_np'):
        return Vec4_np(self.xyzw + other.xyzw, self.c)

    def __sub__(self, other: 'Vec4_np'):
        return Vec4_np(*(self.xyzw - other.xyzw), self.c)

    def __mul__(self, other: float):
        return Vec4_np(*(self.xyzw * other), self.c)

    def __truediv__(self, other: float):
        #check if other is 0
        if other == 0:
            raise ZeroDivisionError("division by zero")
        return Vec4_np(np.divide(self.xyzw, other), self.c)

    def __str__(self):
        return f"({self.xyzw[0]}, {self.xyzw[1]}, {self.xyzw[2]}, {self.xyzw[3]}, {self.c})"        

class Vec4Utils:
    @staticmethod
    def matrix_dot_vector_np(matrix, vector: Vec4_np):
        return np.dot(matrix, vector.xyzw)
        
    #multiply a matrix by another matrix with numpy
    @staticmethod
    def matrix_multiply_matrix(matrix1, matrix2):
        return np.matmul(matrix1, matrix2)

class Projections:
    def look_at(points: list[Vec4_np], eye: Vec4_np, target: Vec4_np, up: Vec4_np):
        # Calculate the forward vector
        forward = (target - eye).normalize_new()

        if True:
            #Calculate the new up vector
            a = forward * up.dot(forward)
            new_up = (up - a).normalize_new()
            up = new_up
            # right = forward.cross(up).normalize_new()
            right = forward.cross(up).normalize_new()
        else:
            # Calculate the right vector
            right = forward.cross(up).normalize_new()
            
            # Calculate the up vector
            up = right.cross(forward).normalize_new()
        # Create the look-at matrix
        rx, ry, rz, _ = right.xyzw
        ux, uy, uz, _ = up.xyzw
        fx, fy, fz, _ = forward.xyzw

        look_at_matrix = [
            [rx, ry, rz, 0],
            [ux, uy, uz, 0],
            [-fx, -fy, -fz, 0],
            [0, 0, 0, 1]
        ]

        return np.array(look_at_matrix)

    @staticmethod
    def perspective_projection(right, left, top, bottom, far, near):
        perspective_matrix = [
            [2 * near / (right - left), 0, (right + left) / (right - left), 0],
            [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0]
        ]

        return np.array(perspective_matrix)


    #sepate the orthographic projection into orthographic and isometric
    @staticmethod
    def orthographic_projection(right, left, top, bottom, near, far):
        Sx = 2 / (right - left)
        Sy = 2 / (top - bottom)
        Sz = -2 / (far - near)
        Tx = - (right + left) / (right - left)
        Ty = - (top + bottom) / (top - bottom)
        Tz = - (far + near) / (far - near)

        projection_matrix = [
            [Sx, 0, 0, Tx],
            [0, Sy, 0, Ty],
            [0, 0, Sz, Tz],
            [0, 0, 0, 1]
        ]
        
        return np.array(projection_matrix)
    
    def isometric_projection():
        Sx = 1 / sqrt(3)
        Sy = 1 / sqrt(3)
        Sz = 1 / sqrt(3)

        projection_matrix = [
            [Sx, 0, 0, 0],
            [0, Sy, 0, 0],
            [0, 0, Sz, 0],
            [0, 0, 0, 1]
        ]

        return np.array(projection_matrix)

    @staticmethod
    def scale_matrix(Sx, Sy, Sz):
        scaling_matrix = [
            [Sx, 0, 0, 0],
            [0, Sy, 0, 0],
            [0, 0, Sz, 0],
            [0, 0, 0, 1]
        ]

        return np.array(scaling_matrix)

    @staticmethod
    def rotate_x_matrix(angle_degrees: float):
        a = radians(angle_degrees)  # Convert angle from degrees to radians
        rotation_matrix = [
            [1, 0, 0, 0],
            [0, cos(a), -sin(a), 0],
            [0, sin(a), cos(a), 0],
            [0, 0, 0, 1]
        ]
        
        return np.array(rotation_matrix)

    @staticmethod
    def rotate_y_matrix(angle_degrees: float):
        a = radians(angle_degrees)  # Convert angle from degrees to radians
        rotation_matrix = [
            [cos(a), 0, sin(a), 0],
            [0, 1, 0, 0],
            [-sin(a), 0, cos(a), 0],
            [0, 0, 0, 1]
        ]
        
        return np.array(rotation_matrix)

    @staticmethod
    def rotate_z_matrix(angle_degrees: float):
        a = radians(angle_degrees)  # Convert angle from degrees to radians
        rotation_matrix = [
            [cos(a), -sin(a), 0, 0],
            [sin(a), cos(a), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        
        return np.array(rotation_matrix)

    @staticmethod
    def translate(points: list[Vec4_np], tx: float, ty: float, tz: float):
        #summing directly to the points
        for vec in points:
            vec.xyzw += np.array([tx, ty, tz, 0])

    @staticmethod
    def translate_point(vec: Vec4_np, tx: float, ty: float, tz: float):
        #vec.xyzw += np.array([tx, ty, tz, 0])
        vec.xyzw = np.add(vec.xyzw, np.array([tx, ty, tz, 0]))

    # define a translate method to move Vec4_np points tx, ty and tz units
    @staticmethod
    def translation_matrix(tx: float, ty: float, tz: float):
        translation_matrix = [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ]

        return  np.array(translation_matrix)

    
    @staticmethod
    def apply_transformation(points: list[Vec4_np], matrix):
        # Apply the given transformation matrix to each point
        for vec in points:
            vec.xyzw = Vec4Utils.matrix_dot_vector_np(matrix, vec)

    @staticmethod
    def apply_transformation_point(vec: Vec4_np, matrix):
        # Apply the given transformation matrix to each point
        vec.xyzw = Vec4Utils.matrix_dot_vector_np(matrix, vec)

class PointAssigner:
    def __init__(self, lists, is_copy=False):
        self.z_max = float('-inf')
        self.z_min = float('inf')
        self.x_max = float('-inf')
        self.y_max = float('-inf')
        self.x_min = float('inf')
        self.y_min = float('inf')
        self.x_len = 0
        self.y_len = 0
        self.lists = []
        self.original_points = []
        self.points = None
        if not is_copy:
            self.assign_points(lists)
            self.size_of_lists(lists)
            self.center_xy_at_origin()
            self.copy_points()

    def copy(self):
        new_assigner = PointAssigner([], True)  # Create a new instance of PointAssigner
        #copy values to new instance not references
        new_assigner.x_max = self.x_max
        new_assigner.y_max = self.y_max
        new_assigner.z_max = self.z_max
        new_assigner.x_min = self.x_min
        new_assigner.y_min = self.y_min
        new_assigner.z_min = self.z_min
        new_assigner.x_len = self.x_len
        new_assigner.y_len = self.y_len
        new_assigner.lists = self.lists
        new_assigner.original_points = self.original_points
        #copy instances of points to new instances
        new_assigner.points = [Vec4_np(point.get_x(), point.get_y(), point.get_z(), point.get_w(), point.get_rgba()) for point in self.points]
        
    def lists_equal(self, lists):
        if not lists:
            exit("File is empty")
        length = len(lists[0]) 
        for y, list in enumerate(lists): #not using slicing so I can render a single line
            if len(list) != length:
                return y
        return None

    def size_of_lists(self, lists):
        if var := self.lists_equal(lists):
            amount = "less" if len(lists[var]) < len(lists[0]) else "more"
            s = "s" if var > 2 else ""
            exit(f"line {str(var)} [0-indexed] has {amount} elements than the previous line{s}.")
        else:
            self.x_len = len(lists[0])
            self.y_len = len(lists)

    def extract_color(self, hex):
        try:
            if hex.startswith("0x") and 2 < len(hex) < 11: # check if value has more than the 0x and less than 0xAAAAAAAA
                num = int(hex[2:], 16) # no need to check if value  is between 0 and 4294967295 as any other value will raise a ValueError
                a = 255 if len(hex) <= 8 else (num >> 24) & 0xFF #if the number has no alpha value (0x[AA]RRGGBB) make `a = 255`
                r = (num >> 16) & 0xFF
                g = (num >> 8) & 0xFF
                b = num & 0xFF
                return [r, g, b, a]
            else:
                raise MyWarning("wrong format. Should be '...z z z...' or '...z z,0x[[[AA]RR]GG]BB z...'\nWhere '0xAARRGGBB' are hexadecimal values 0-f\nAnd values in [] are optional")
        except ValueError:
            raise MyWarning("wrong format. Color should be hexadecimal [0-f]")
        
    def no_color(self, z):
        if -10 < z < 10:
            return self.extract_color("0x9A1F6A")
        elif z < -10:
            return self.extract_color("0xFFFFFF")
        else:
            return self.extract_color("0xF3B03D")

    def parse_point(self, point):
        try:
            parts = point.split(",")
            if len(parts) == 1:
                return float(parts[0]), None
            elif len(parts) == 2:
                z = float(parts[0])
                color = self.extract_color(parts[1])
                return z, color
        except ValueError as e:
            raise MyWarning(f"Something didn't parse. Was the value an not numeric or too big?\nValueError: {e}")

    def assign_points(self, lists):
        y_list = []
        for y, l in enumerate(lists):
            x_list = []
            for x, point in enumerate(l):
                try:
                    z, color = self.parse_point(point)
                    if color is None:
                        color = self.no_color(z)
                    if z > self.z_max:
                        self.z_max = z
                    if z < self.z_min:
                        self.z_min = z
                    x_list.append(z)
                    self.original_points.append(Vec4_np(float(x), float(y), float(z), 1, color))
                except MyWarning as e:
                    exit(f"Error on line {str(y)}, item {str(x)}: {e}")
            y_list.append(x_list)
        self.lists = y_list

    def get_z_max_min(self):
        z_values = [point.get_z() for point in self.points]
        self.z_max = max(z_values)
        self.z_min = min(z_values)
        return self.z_max, self.z_min

    def get_points(self):
        return self.points
    
    def get_x_y_len(self):
        return self.x_len, self.y_len

    def get_point_at_position(self, x, y):
        i = y * self.x_len + x
        return self.points[i]
    
    def get_center_point(self):
        return self.get_point_at_position(self.x_len // 2, self.y_len // 2)

    def get_xyz_max_min_values(self):
        for point in self.points:
            x, y, z, _ = point.get_xyzw()
            self.x_max = max(self.x_max, x)
            self.y_max = max(self.y_max, y)
            self.z_max = max(self.z_max, z)
            self.x_min = min(self.x_min, x)
            self.y_min = min(self.y_min, y)
            self.z_min = min(self.z_min, z)
        return self.x_max, self.y_max, self.z_max, self.x_min, self.y_min, self.z_min

    def get_xy_center_point(self):
        total_x, total_y = 0, 0
        length = len(self.original_points)
        for point in self.original_points:
            total_x += point.get_x()
            total_y += point.get_y()
        center_x = total_x / length
        center_y = total_y / length
        return center_x, center_y

    def old_center_xy_at_origin(self):
        center_x, center_y = self.get_xy_center_point()
        for point in self.points:
            point.set_x(point.get_x() - center_x)
            point.set_y(point.get_y() - center_y)

    def center_xy_at_origin(self):
        center_x, center_y = self.get_xy_center_point()
        Projections.translate(self.original_points, -center_x, -center_y, 0)

    def lists_get_x_y_value(self, x, y):
        return self.lists[y][x]

    def copy_points(self):
        self.points = [Vec4_np(point.get_x(), point.get_y(), point.get_z(), point.get_w(), point.get_rgba()) for point in self.original_points]

    def redo(self):
        if self.points is not None:
            #delete every instance of points
            #make sure it is out of memory
            for point in self.points:
                del point
            self.points = []
        self.copy_points()

class CQPainter(QPainter):
    def drawLineGradient(self, x1, y1, x2, y2, color1, color2, width=1):
        start_point = QPoint(x1, y1)
        end_point = QPoint(x2, y2)

        gradient = QLinearGradient(start_point, end_point)
        gradient.setColorAt(0, QColor(*color1))
        gradient.setColorAt(1, QColor(*color2))

        pen = QPen()
        pen.setBrush(gradient)
        pen.setWidth(width)
        self.setPen(pen)
        self.drawLine(start_point, end_point)

class MainWindow(QMainWindow):
    def __init__(self, points_assigner: PointAssigner):
        super().__init__()
        self.pa = points_assigner
        self.list = self.pa.points

        self.label = QLabel()
        self.canvas = QPixmap(800, 600)
        self.canvas.fill(QColor("#333333"))
        self.label.setPixmap(self.canvas)
        self.setCentralWidget(self.label)

        self.scene_width = self.label.pixmap().width()
        self.scene_height = self.label.pixmap().height()

        self.eye = Vec4_np(10.0, 10.0, 10.0, 1.0) #camera position
        self.target = Vec4_np(0.0, 0.0, 0.0, 1.0) #look at the center of the scene
        self.up = Vec4_np(0.0, 0.0, -1.0, 1.0) #camera up vector 1,1,0 also works

        #hold bbox values
        self.right = 0.05
        self.left = -self.right
        self.top = self.right
        self.bottom = -self.top
        self.near = 1
        self.far = 100

        #hold rotation values
        self.x_angle = 0
        self.y_angle = 0
        self.z_angle = 0
        self.zoom = 1
        self.proj = "iso"
        self.Sz = 1
        self.menu = True

        self.space_counter = 0

        self.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_P, Qt.NoModifier))

    def keyPressEvent(self, event):
        key=event.key()
        self.pa.redo()
        self.list = self.pa.points
        
        if key == Qt.Key_Escape:
            self.close()
        speed = 1
        if key == Qt.Key_R:
            self.x_angle += 5
        if key == Qt.Key_T:
            self.x_angle -= 5
        if key == Qt.Key_F:
            self.y_angle += 5
        if key == Qt.Key_G:
            self.y_angle -= 5
        if key == Qt.Key_V:
            self.z_angle += 5
        if key == Qt.Key_B:
            self.z_angle -= 5

        # k-l for zoom in and out
        if key == Qt.Key_K:
            self.zoom += 0.1
        if key == Qt.Key_L:
            self.zoom -= 0.1

        # Q for the menu
        if key == Qt.Key_Q:
            if self.menu:
                self.menu = False
            else:
                self.menu = True
        
        #switch between orthographic(o), isometric(i) and perspective(p) projection
        if key == Qt.Key_O:
            self.proj = "ort"
        if key == Qt.Key_I:
            self.proj = "iso"
            self.eye.set_xyzw(10, 10, 10, 1)
            self.up.set_xyzw(0, 0, -1, 1)
            self.x_angle = 0
            self.y_angle = 0
            self.z_angle = 0
        if key == Qt.Key_P:
            self.proj = "per"

        #scale z axis
        if key == Qt.Key_Z:
            self.Sz += 0.1
        if key == Qt.Key_X:
            self.Sz -= 0.1

        if key == Qt.Key_C:
            self.space_counter -= 1
        if key == Qt.Key_Space:
            self.space_counter += 1


        self.project()
        r_matrix = Projections.rotate_x_matrix(self.x_angle) @ Projections.rotate_y_matrix(self.y_angle) @ Projections.rotate_z_matrix(self.z_angle) @ Projections.scale_matrix(1, 1, self.Sz)
        l_matrix = Projections.look_at(self.list, self.eye, self.target, self.up)
        s_matrix = []
        if self.proj == "ort":
            s_matrix = Projections.scale_matrix(self.zoom, self.zoom, self.zoom)
            p_matrix = Projections.orthographic_projection(self.right, self.left, self.top, self.bottom, self.near, self.far)
        elif self.proj == "iso":
            s_matrix = Projections.scale_matrix(50 * self.zoom, 50 * self.zoom, 50 * self.zoom)
            p_matrix = Projections.isometric_projection()
        else:
            s_matrix = Projections.scale_matrix(100 * self.zoom, 100 * self.zoom, 100 * self.zoom)
            p_matrix = Projections.perspective_projection(self.right, self.left, self.top, self.bottom, self.far, self.near)
        t_matrix = Projections.translation_matrix(self.scene_width // 2, self.scene_height // 2, 0)
        
        if self.proj == "iso" or self.proj == "ort":
            test = t_matrix @ p_matrix @ s_matrix @ l_matrix @ r_matrix
        else:
            test = s_matrix @ p_matrix @ l_matrix @ r_matrix

        for point in self.list:
            point.xyzw = Vec4Utils.matrix_dot_vector_np(test, point)
            if self.proj == "per":
                Projections.translate_point(point, self.scene_width // 2, self.scene_height // 2, 0)

        
        self.draw_view()
        if self.menu:
            self.draw_menu()

        
    def draw_menu(self):
        #draw a rectangle on the canvas 
        rectangle_color = QColor(*self.pa.extract_color("0xc3c3c3"))
        rectangle_position = (0, 0)
        rectangle_size = (250, self.scene_height)
        font = QFont("Consolas", 12)
        text_color = QColor(0, 0, 0)
        painter = CQPainter(self.label.pixmap())

        painter.setBrush(rectangle_color)
        painter.drawRect(*rectangle_position, *rectangle_size)
        painter.setFont(font)
        painter.setPen(text_color)
        text_items = ["Toggle Menu: Q", "rotate x: R/T", "rotate y: F/G", "rotate z: V/B", "zoom in/out: K/L", "scale z: Z/X", "", "Projections:", "orthographic: O", "isometric: I", "perspective: P"]
        for i, text in enumerate(text_items):
            painter.drawText(10, 50 + i * 20, text)

    def project(self):
        w = self.scene_width
        h = self.scene_height
        fov = radians(90.0)
        aspect_ratio = w / h

        if self.proj == "ort":
            self.right = 0.03
            self.left = -self.right
            self.top = self.right
            self.bottom = -self.top
            self.near = 1
            self.far = 100
        else:
            self.right = aspect_ratio * tan(fov / 2) + 1
            self.left = -self.right
            self.top = aspect_ratio / tan(fov / 2) + 1
            self.bottom = -self.top
            self.near = 1
            self.far = 1000
           

    #draw but only draws the lines that are in the view
    def draw_view(self):
        self.label.setPixmap(self.canvas)
        painter = CQPainter(self.label.pixmap())
        # painter.setRenderHint(QPainter.Antialiasing)
        ix, iy = self.pa.get_x_y_len()
        for y in range(iy):
            for x in range(ix):
                point = self.pa.get_point_at_position(x, y)
                if x + 1 < ix:
                    next_point = self.pa.get_point_at_position(x + 1, y)
                    if self.is_in_view(point, next_point):
                        painter.drawLineGradient(round(point.get_x()), round(point.get_y()), round(next_point.get_x()), round(next_point.get_y()), point.get_rgba(), next_point.get_rgba())
                #do the same for the bottom adjacent point
                if y + 1 < iy:
                    next_point = self.pa.get_point_at_position(x, y + 1)
                    if self.is_in_view(point, next_point):
                        painter.drawLineGradient(round(point.get_x()), round(point.get_y()), round(next_point.get_x()), round(next_point.get_y()), point.get_rgba(), next_point.get_rgba())
        painter.end()
        self.label.update()

    def is_in_view(self, point1, point2):
        #check if point1 or point2 is in view
        if self.is_point_in_view(point1):
            return True
        if self.is_point_in_view(point2):
            return True
        #check if the line between point1 and point2 is in view
        if self.is_line_in_view(point1, point2):
            return True
        return False
    
    def is_point_in_view(self, point):
        #check if point is in view
        if 0 < point.get_x() < self.scene_width and 0 < point.get_y() < self.scene_height:
            return True
        return False
    
    def is_line_in_view(self, point1, point2):
        #check if line between point1 and point2 is in view
        if self.is_point_in_view(point1) and self.is_point_in_view(point2):
            return True
        #check if line is in view
        if self.is_line_in_view_x(point1, point2):
            return True
        if self.is_line_in_view_y(point1, point2):
            return True
        return False
    
    def is_line_in_view_x(self, point1, point2):
        #check if line is in view on x axis
        if 0 < point1.get_x() < self.scene_width and 0 < point2.get_x() < self.scene_width:
            return True
        return False
    
    def is_line_in_view_y(self, point1, point2):
        #check if line is in view on y axis
        if 0 < point1.get_y() < self.scene_height and 0 < point2.get_y() < self.scene_height:
            return True
        return False
        
def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def T():
    return time.time()

def is_fdf(filename):
    """
    checks if filename is a valid file or not

    Parameters:
    filename (str): name of a file

    returns:
    bool: True if filename is a valid file False if not
    """
    if path.isfile(filename) and filename.lower().endswith(('.fdf')):
        return True
    else:
        return False


def lines_to_list(lines):
    """
    removes trailing newline and splits line into individual itemms

    Parameters:
    lines (list): list of strings containing z values (and hex color)

    returns:
    list: list of lists containing z values(and hex color)
    """
    try:
        list_lists = []
        for line in lines:
            line = line.replace("\n", "")
            list = line.split(" ")
            list = [item for item in list if item != ''] # removes empty entries when you have consecutive spaces
            list_lists.append(list)
        return list_lists
    except (ValueError):
        return None


def open_file():
    if len(sys.argv) != 2:
        exit("Wrong arguments.\nusage: [python] PyFdF.py <map.fdf>")
    if not is_fdf(sys.argv[1]):
        exit("Error opening file or wrong file ext")
    with open(sys.argv[1]) as file:
        lines = file.readlines()
    if lists_of_nums := lines_to_list(lines):
        return lists_of_nums
    else:
        exit("Error processing file")


def main():
    lists = open_file()
    points = PointAssigner(lists)

    app = QApplication([])
    app.setStyle('Fusion')
    draw_points = MainWindow(points)
    draw_points.show()
    sys.exit(app.exec_())
   

if __name__ == "__main__":
    main()