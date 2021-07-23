# -*- mode: python; indent-tabs-mode: nil; tab-width: 4 -*-
# vim: set tabstop=4 shiftwidth=4 expandtab:

#
# Copyright (C) 2021 Mathias Meuleman
#
# Based on the work of Christopher Dalitz 2005
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#  
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

# import exceptions

from gamera import toolkit
from gamera.args import *
# from gamera.args import Args, Int
from gamera.core import ONEBIT, RGB, RGBPixel, Point, Image
from gamera.gui import has_gui


##############################################################################
#
# data structures representing a single staff line
#
class StafflineAverage:
    """Represents a staff line as a single y-value and contains the
following positional information as public properties:

  *left_x*, *right_x*:
    x-position of left and right edge
  *average_y*:
    average y-position of the staffline
  *variance_y*:
    variance in the y-position of the staffline
"""
    def __init__(self):
        """The constructor has no arguments. All values are accessed
directly.
"""
        self.left_x = None
        self.right_x = None
        self.average_y = None
        self.variance_y = 0

    # conversion functions
    def to_average(self):
        """Converts to ``StafflineAverage``. Thus simply returns *self*."""
        return self

    def to_polygon(self, tolerance=0):
        """Converts to ``StafflinePolygon``.

The optional parameter *tolerance* is offered for compatibility to
``StafflineSkeleton``, but has no effect on the return value.
"""
        pg = StafflinePolygon()
        pg.vertices += [ Point(self.left_x, self.average_y),
                         Point(self.right_x, self.average_y) ]
        return pg

    def to_skeleton(self):
        """Converts to ``StafflineSkeleton``."""
        sk = StafflineSkeleton()
        sk.left_x = self.left_x
        sk.y_list = [self.average_y] * (self.right_x - self.left_x + 1)
        return sk


class StafflinePolygon:
    """Represents a staff line as a polygon and contains the list of
vertex points as public property:

  *vertices*:
    a list of the vertices. Each point is stored as Gamera's data type *Point*
"""
    def __init__(self):
        """The constructor has no arguments. All values are accessed
directly.
"""
        self.vertices = []

    # conversion functions
    def to_average(self):
        """Converts to ``StafflineAverage``."""
        av = StafflineAverage()
        if self.vertices:
            av.left_x = self.vertices[0].x
            av.right_x = self.vertices[-1].x
            if len(self.vertices) == 1:
                av.average_y = self.vertices[0].y
                av.variance_y = 0
            else:
                # although mean and variance could be computed in a single
                # loop, we use two loops to avoid overflow in sum(y**2)
                lastp = None
                n = 0
                y = 0.0
                for p in self.vertices:
                    if lastp and (p.x > lastp.x):
                        y += 0.5 * (p.y + lastp.y) * (p.x - lastp.x)
                        n += (p.x - lastp.x)
                    lastp = p
                avgy = int(y / n)
                lastp = None
                y2 = 0.0
                for p in self.vertices:
                    if lastp and (p.x > lastp.x):
                        dy = float(p.y - lastp.y)
                        by = lastp.y - avgy - lastp.x * dy / (p.x - lastp.x)
                        y2 += dy**2 * (p.x**2 + lastp.x * (lastp.x + p.x)) / (3.0*(p.x - lastp.x))
                        y2 += by * (dy * (p.x + lastp.x) + by * (p.x - lastp.x))
                    lastp = p
                av.average_y = avgy
                av.variance_y = y2/n
        return av

    def to_polygon(self, tolerance=0):
        """Converts to ``StafflinePolygon``. Thus simply returns *self*.

The optional parameter *tolerance* is offered for compatibility to
``StafflineSkeleton``, but has no effect on the return value.
"""
        return self

    def to_skeleton(self):
        """Converts to ``StafflineSkeleton``."""
        sk = StafflineSkeleton()
        if self.vertices:
            sk.left_x = self.vertices[0].x
            sk.y_list.append(self.vertices[0].y)
            lastp = None
            for p in self.vertices:
                if lastp and (p.x > lastp.x):
                    dy = float(p.y - lastp.y) / (p.x - lastp.x)
                    x = lastp.x + 1
                    y = lastp.y
                    while x <= p.x:
                        y += dy
                        sk.y_list.append(int(y))
                        x += 1
                lastp = p
        return sk


class StafflineSkeleton:
    """Represents a staff line as a skeleton, i.e. a one point thick
continuous path. The positional information of the path is stored in the
following public properties:

  *left_x*:
    left most x-position of the sekeleton
  *y_list*:
    list of subsequent y-positions from left to right

Consequently the right most x-position of the skeleton is
*left_x + len(y_list)*.
"""
    def __init__(self):
        """The constructor has no arguments. All values are accessed
directly.
"""
        self.left_x = None
        self.y_list = []

    # conversion functions
    def to_average(self):
        """Converts to ``StafflineAverage``."""
        av = StafflineAverage()
        av.left_x = self.left_x
        n = len(self.y_list)
        av.right_x = self.left_x + n - 1
        if n == 0:
            return av
        # although mean and variance could be computed in a single
        # loop, we use two loops to avoid overflow in sum(y**2)
        sumy = 0
        for y in self.y_list:
            sumy += y
        avgy = float(sumy) / n
        sumy2 = 0.0
        for y in self.y_list:
            sumy2 += (y - avgy)**2
        av.average_y = int(avgy + 0.5)
        av.variance_y = sumy2 / n
        return av

    def to_polygon(self, tolerance=2):
        """Converts to ``StafflinePolygon``. Not yet implemeted.

The parameter *tolerance* controls how accurate the polygonal approximation
is.
"""
        raise RuntimeError("Conversion not yet implemented")

    def to_skeleton(self):
        """Converts to ``StafflineSkeleton``. Thus simply returns *self*."""
        return self


##############################################################################
#
#
class StaffFinder:
    """Virtual class providing a unified interface to staff line finding
from a music/tablature image without their removal.
For typical usage see the `user's manual`__.

.. __: usermanual.html

Important properties:

  *image*:
    The onebit image in which the staff lines are to be found.
    You might need to access this property, if the algorithm modifies
    this image during the staff finding process (e.g. by a line
    rectifying process).

  *staffline_height*, *staffspace_height*:
    Line thickness and distance of adjacent staff lines.

  *linelist*:
    Data structure in which the positional information is stored.
    It contains a nested list of ``StafflinAverage`` or ``StafflinePolygon``
    or ``StafflineSkeleton`` objects; each sublist represents a stave system.
    In order to be independant of the actual internal representation do not
    read this variable directly from outside the class, but use the methods
    ``get_average()``, ``get_polygon()`` or ``get_skeleton()``.

.. note:: This class cannot be used itself. You must use one of the
   derived classes which actually implement staff finding algorithms.

:Author: Christoph Dalitz
"""
    ######################################################################
    # __init__
    #
    # definition of variables
    #
    def __init__(self, img, staffline_height=0, staffspace_height=0):
        """Constructor of StaffFinder. This base constructor should be
called by all derived classes.

Signature:

  ``__init__(image, staffline_height=0, staffspace_height=0)``

with

  *image*:
    Onebit or greyscale image of which the staves are to be found.
    StaffFinder creates a copy of the given image, so that the original
    image is not altered. The staff finding and possible additional operations
    will be done on ``StaffFinder.image``

  *staffline_height*:
    average height of a single black staff line. When zero is given,
    StaffFinder computes it itself (recommended, unless you know exactly
    what you are doing).

  *staffspace_height*:
    average distance between adjacent lines of the same staff. When zero is
    given, StaffFinder computes it itself (recommended, unless you know exactly
    what you are doing).
"""
        if img.data.pixel_type == ONEBIT:
            self.image = img.image_copy()
        else:
            self.image = img.to_onebit()
        self.staffline_height = staffline_height
        self.staffspace_height = staffspace_height

        # height of a single staffline
        if 0 == self.staffline_height:
            self.staffline_height = self.image.most_frequent_run('black', 'vertical')

        # space between 2 stafflines in the same system
        if 0 == self.staffspace_height:
            self.staffspace_height = self.image.most_frequent_run('white', 'vertical')

        # data structure for positional information
        # contains a nested list of StafflinAverage or
        # StafflinePolygon or StafflineSkeleton objects
        self.linelist = []

        # some variables for caching GUI information
        # (avoids their frequent recomputation)
        self.averagelist = None  # average staff line y-positions
        self.rgbstaves = None   # RGB image with found staves colored

        # arguments of find_staves method for GUI dialog
        self.find_staves_args = Args([Int("num_lines", default=0)])

    ######################################################################
    # find_staves
    #
    def find_staves(self, num_lines=0):
        """Finds staff lines from a music/tablature image.

This method is virtual and must be implemented in every derived class.
Each implementation must at least support the following keyword argument:

  *num_lines*:
    Number of lines within one staff. In general it can be hard to guess this
    number automatically; if the staff removal algorithm supports this, a value
    *num_lines=0* can be used to indicate that this number should be guessed
    automatically.

Additional algorithm specific keyword are possible of course. If
*num_lines* cannot be chosen in the algorithm, this option must
be offered nevertheless (albeit with no effect).
"""
        raise RuntimeError("Module StaffFinder (virtual class): Method not implemented.")

    ######################################################################
    # get_average
    #
    def get_average(self):
        """Returns the average y-positions of the staff lines.
When the native type of the StaffFinder implementation is not
`StafflineAverage`_, the average values are computed.

The return value is a nested list of `StafflineAverage`_ where each
sublist represents one stave group.

.. _StafflineAverage: gamera.toolkits.musicstaves.stafffinder.StafflineAverage.html
"""
        if self.linelist:
            returnlist = []
            for staff in self.linelist:
                returnlist.append([s.to_average() for s in staff])
            return returnlist
        else:
            return []

    ######################################################################
    # get_skeleton
    #
    def get_skeleton(self):
        """Returns the skeleton of the staff lines.
When the native type of the StaffFinder implementation is not
`StafflineSkeleton`_, the skeleton is computed.

The return value is a nested list of `StafflineSkeleton`_ where each
sublist represents one stave group.

.. _StafflineSkeleton: gamera.toolkits.musicstaves.stafffinder.StafflineSkeleton.html
"""
        if self.linelist:
            returnlist = []
            for staff in self.linelist:
                returnlist.append([s.to_skeleton() for s in staff])
            return returnlist
        else:
            return []

    ######################################################################
    # get_polygon
    #
    def get_polygon(self):
        """Returns a polygonal approximation of the staff lines.
When the native type of the StaffFinder implementation is not
`StafflinePolygon`_, the polygonal approximation is computed.

The return value is a nested list of `StafflinePolygon`_ where each
sublist represents one stave group.

.. _StafflinePolygon: gamera.toolkits.musicstaves.stafffinder.StafflinePolygon.html
"""
        if self.linelist:
            returnlist = []
            for staff in self.linelist:
                returnlist.append([s.to_polygon() for s in staff])
            return returnlist
        else:
            return []


    ######################################################################
    # show_result
    #
    def show_result(self, highlight_groups=True):
        """Returns an RGB image with the found staff lines highlighted.

Parameters:

  *highlight_groups*:
    when *True* (default), each stave system is underlayed in a different
    color. When false, only the edge points are marked in different colors,
    which is less visible for skeleton or average staff line representation.
"""
        rgb = Image(self.image, RGB)
        if highlight_groups:
            # highlight staff systems
            for staffno, staff in enumerate(self.linelist):
                if not staff:
                    continue
                staffcolor = RGBPixel(190 + (11*staffno) % 66,
                                      190 + (31*(staffno + 1)) % 66,
                                      190 + (51*(staffno + 2)) % 66)
                topline = staff[0].to_average()
                botline = staff[-1].to_average()
                leftx = min([topline.left_x, botline.left_x])
                rightx = max([topline.right_x, botline.right_x])
                topy = topline.average_y
                boty = botline.average_y
                border = max([self.staffspace_height/2, self.staffline_height*4])
                if leftx - border > 0:
                    leftx -= border
                if rightx + border < rgb.ncols:
                    rightx += border
                if topy - border > 0:
                    topy -= border
                if boty + border < rgb.nrows:
                    boty += border
                rgb.draw_filled_rect((leftx, topy), (rightx, boty), staffcolor)

        # draw original image
        rgb.highlight(self.image, RGBPixel(0, 0, 0))

        for staffno, staff in enumerate(self.linelist):
            if not staff:
                continue
            if highlight_groups:
                staffcolor = RGBPixel(255, 0, 0)
            else:
                staffcolor = RGBPixel((53 * staffno) % 256, (111*(staffno+1)) % 256, (111*(staffno+2)) % 256)

            for line in staff:
                if isinstance(line, StafflinePolygon):
                    lastp = None
                    for p in line.vertices:
                        rgb.draw_marker(p, self.staffline_height*2, 3, staffcolor)
                        if lastp:
                            rgb.draw_line(lastp, p, RGBPixel(255, 0, 0))
                        lastp = p
                elif isinstance(line, StafflineAverage):
                    rgb.draw_line(Point(0, line.average_y), Point(rgb.ncols, line.average_y), RGBPixel(255, 0, 0))
                    rgb.draw_marker(Point(line.left_x, line.average_y), self.staffline_height*2, 3, staffcolor)
                    rgb.draw_marker(Point(line.right_x, line.average_y), self.staffline_height*2, 3, staffcolor)
                elif isinstance(line, StafflineSkeleton):
                    # doing this in Python might be too slow,
                    # implement it in C++ later
                    x = line.left_x
                    for y in line.y_list:
                        rgb.set(Point(x, y), RGBPixel(255, 0, 0))
                        x += 1
                    if len(line.y_list) > 0:
                        rgb.draw_marker(Point(line.left_x, line.y_list[0]), self.staffline_height*2, 3, staffcolor)
                        rgb.draw_marker(Point(x-1, line.y_list[-1]), self.staffline_height*2, 3, staffcolor)
        return rgb
