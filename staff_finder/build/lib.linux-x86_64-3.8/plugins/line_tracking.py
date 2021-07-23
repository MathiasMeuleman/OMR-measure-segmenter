# -*- mode: python; indent-tabs-mode: nil; tab-width: 4 -*-
# vim: set tabstop=4 shiftwidth=4 expandtab:

#
# Copyright (C) 2005 Thomas Karsten, Christoph Dalitz and Florian Pose
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

#----------------------------------------------------------------

from gamera.plugin import *
import _line_tracking

#----------------------------------------------------------------

class get_staff_skeleton_list(PluginFunction):
    """Returns skeleton data of staffline candidates.

In order to find staffline candidates, long quasi black runs are extracted
with extract_filled_horizontal_black_runs_. The resulting filaments are
vertically thinned with thinning_v_to_skeleton_list_.

.. _extract_filled_horizontal_black_runs: musicstaves.html#extract-filled-horizontal-black-runs

.. _thinning_v_to_skeleton_list: musicstaves.html#thinning-v-to-skeleton-list

Arguments:

  *windowwidth*, *blackness*:
    Input arguments passed on to `extract_filled_horizontal_black_runs`__.

.. __: musicstaves.html#extract-filled-horizontal-black-runs

  *staffline_height*:
    Input argument passed on to `thinning_v_to_skeleton_list`__. This is
    half the threshold for splitting a vertical black run into more than
    one skeleton points.

.. __: musicstaves.html#thinning-v-to-skeleton-list

Return value:

  *skeleton_list*:
    A nested list, where each sublist represents a staffline candidate
    skeleton as an array of two elements: the first element is the leftmost
    x position, the second element is a list of subsequent y-values.

    The returned skeleton list is sorted from top to bottom and from left
    to right.
"""
    category = "MusicStaves/Line_tracking"
    self_type = ImageType([ONEBIT, GREYSCALE])
    args = Args([Int('staffline_height'),\
            Int('windowwidth'),\
            Int('blackness', range=(1, 100), default=75)])
    return_type = Class("skeleton_list")
    author = "Thomas Karsten and Christoph Dalitz"
    pure_python = 1

    def __call__(self, staffline_height, windowwidth, blackness=75):

        class sort_class:
            def __init__(self, staffline_height):
                self.__staffline_height=staffline_height
                self.max_length=0

            # sort a and b by
            #     1. y-values
            # and if the y-values are within a specified range by
            #     2. x-values
            # additionally, curved lines are taken care of
            def sort_skeleton_list(self, a, b):
                # first: get the maximum length of a staffline
                if len(a[1]) > self.max_length:
                    self.max_length=len(a[1])
                if len(b[1]) > self.max_length:
                    self.max_length=len(b[1])

                # a line is broken: compare the last
                # y-position of 'a' with the first y-position
                # of 'b'
                if a[0] < b[0] and a[0]+len(a[1]) <= b[0]:
                    if a[1][-1] < b[1][0]+\
                            self.__staffline_height\
                            and\
                            a[1][-1] > b[1][0]-\
                            self.__staffline_height:
                        return int(a[0]-b[0])

                # 'b' is before 'a': compare the last
                # y-position of 'b' with the first y-position
                # of 'a'
                elif a[0] > b[0] and b[0]+len(b[1]) <= a[0]:
                    if b[1][-1] < a[1][0]+\
                            self.__staffline_height\
                            and\
                            b[1][-1] > a[1][0]-\
                            self.__staffline_height:
                        return int(a[0]-b[0])

                # lines do not seem to belong together, return
                # the difference between their first
                # y-positions
                return int(a[1][0]-b[1][0])

        image=_line_tracking.extract_filled_horizontal_black_runs(\
                self, windowwidth, blackness)
        skeleton_list=_line_tracking.thinning_v_to_skeleton_list(\
                image, staffline_height)

        # sort *all* detected lines and get the maximum length of a
        # line (should usually also be the average length of normal
        # staves)
        sort_obj=sort_class(staffline_height)
        skeleton_list.sort(sort_obj.sort_skeleton_list)
        max=sort_obj.max_length

        return skeleton_list

    __call__ = staticmethod(__call__)

#----------------------------------------------------------------

class extract_filled_horizontal_black_runs(PluginFunction):
    """Extract horizontal lines from an image.

Returns black horizontal runs with a blackness greater than *blackness* within
a horizontal window of width *windowwidth*. In essence this is a combination
of ``fill_horizontal_line_gaps`` and ``filter_narrow_runs(x, 'black')``.
"""
    category = "MusicStaves/Line_tracking"
    self_type = ImageType([ONEBIT, GREYSCALE])
    args = Args([Int('windowwidth'),\
            Int('blackness', range=(1, 100), default=75)])
    return_type = ImageType([ONEBIT])
    author = "Thomas Karsten and Christoph Dalitz"

    def __call__(self, windowwidth, blackness=75):
        return _line_tracking.extract_filled_horizontal_black_runs(\
                self, windowwidth, blackness)
    __call__ = staticmethod(__call__)

#----------------------------------------------------------------

class thinning_v_to_skeleton_list(PluginFunction):
    """Simplified thinning algorithm that only thins in the vertical
direction.

Return value:
  *skeleton_list*:
    Nested list of skeleton data. See `get_staff_skeleton_list`__ for details.

.. __: musicstaves.html#get-staff-skeleton-list

Each vertical black run in the image is thinned down to its middle value,
provided the black run is not too tall. Black runs taller than
2 * *staffline_height* are thinned more than once.
"""
    category = "MusicStaves/Line_tracking"
    self_type = ImageType([ONEBIT])
    args = Args([Int("staffline_height")])
    return_type = Class("skeleton_list")
    author = "Thomas Karsten and Christoph Dalitz"

    def __call__(self, staffline_height):
        return _line_tracking.thinning_v_to_skeleton_list(self,\
                staffline_height)
    __call__ = staticmethod(__call__)

#----------------------------------------------------------------

class skeleton_list_to_image(PluginFunction):
    """Creates an image using a skeleton list as returned by
get_staff_skeleton_list_.

Argument:
  *skeleton_list*:
    Nested list of skeleton data. See `get_staff_skeleton_list`__ for details.

.. __: musicstaves.html#get-staff-skeleton-list

This function only helps to visualize a skeleton list. The image this
function operates on is only needed for determining *ncols* and *nrows*
of the output image. Typically the input image is the original image from
which the skeletons have been extracted.
"""
    category = "MusicStaves/Line_tracking"
    self_type = ImageType(ALL)
    args = Args([Class("skeleton_list")])
    return_type = ImageType(ALL)
    author = "Thomas Karsten"

    def __call__(self, list):
        return _line_tracking.skeleton_list_to_image(self, list)
    __call__ = staticmethod(__call__)

#----------------------------------------------------------------

class follow_staffwobble(PluginFunction):
    """Returns a staffline skeleton that follows a wobbling staffline.

This function is useful when you only approximately know the stafflines
(eg. if you only know the average y position for each staffline). It
follows the staff line by seeking adjacent slithers of approximately
*staffline_height*.

Arguments:

  *staffline_object*:
    A single staffline represented as StafflineAverage_, StafflinePolygon_ or
    StafflineSkeleton_.

.. _StafflineAverage: gamera.toolkits.musicstaves.stafffinder.StafflineAverage.html
.. _StafflinePolygon: gamera.toolkits.musicstaves.stafffinder.StafflinePolygon.html
.. _StafflineSkeleton: gamera.toolkits.musicstaves.stafffinder.StafflineSkeleton.html

  *staffline_height*:
    When zero, it is automatically computed as most frequent white vertical
    runlength.

Return value:

  The (hopefully) more accurate staffline represented as StafflineSkeleton__.

.. __: gamera.toolkits.musicstaves.stafffinder.StafflineSkeleton.html

References:

- D. Bainbridge, T.C. Bell: *Dealing with superimposed objects in
  optical music recognition*. http://www.cs.waikato.ac.nz/~davidb/publications/ipa97/

.. note:: This postprocessing step is only reliable when staff lines
   are absolutely solid and are not interrupted. Otherwise this method
   can result in grossly inaccurate staff lines.
"""
    category = "MusicStaves/Line_tracking"
    self_type = ImageType([ONEBIT])
    args = Args([Class('staffline_object'),\
                 Int('staffline_height', default=0),\
                 Int('debug', default=0)])
    return_type = Class("StafflineSkeleton")
    author = "Christoph Dalitz"

    def __call__(self, staffline_object, staffline_height=0, debug=0):
        if staffline_height == 0:
            staffline_height = self.most_frequent_run('black', 'vertical')
        skel = staffline_object.to_skeleton()
        return _line_tracking.follow_staffwobble(self, skel, staffline_height, debug)
    __call__ = staticmethod(__call__)

#----------------------------------------------------------------

class remove_line_around_skeletons(PluginFunction):
    """Removes a line of a certain thickness around
every skeleton in the given list.

For each skeleton pixel the vertical black runlength around this
pixel is determined. Is it greater than the *threshold*, a vertical
runlength of *staffline_height* is removed; is it smaller, the entire
vertical black runlength is removed.

The black runlength my contain gaps of *max_gap_height* pixels. Even
the skeleton pixel is allowed to be white, if the surrounding gap does
not exceed the maximum gap height."""

    category = None
    self_type = ImageType(ONEBIT)
    return_type = None
    args = Args([Class("skeleton_list"), \
                 Int("staffline_height"), \
                 Int("threshold"), \
                 Int("max_gap_height", default = 0)])
    author = "Florian Pose"

    def __call__(self, skel_list, slh, threshold, gap):

        return _line_tracking.remove_line_around_skeletons( \
            self, skel_list, slh, threshold, gap)

    __call__ = staticmethod(__call__)

#----------------------------------------------------------------

class rescue_stafflines_using_mask(PluginFunction):
    """A T-shaped mask of 4 pixels is placed above every vertical
'slice' of a staff line. If the lower pixel of the mask is
black and at least one of the upper 3 pixels is black, the
vertical runlength of the staff line is rescued (i. e.
copied to the (actual) rescue image).

- *original_image* is the image without the staff lines removed.

- *skeleton*: All skeletons in a nested list.

- *staffline_height*: (self-explaining)

- *threshold*: The two masks are placed, *threshold* pixels apart,
  above and below the skeleton."""

    category = None
    self_type = ImageType(ONEBIT)
    return_type = None
    args = Args([ImageType([ONEBIT], 'original_image'), \
                 Class("skeleton_list"), \
                 Int("staffline_height"), \
                 Int("threshold")])
    author = "Florian Pose"

    def __call__(self, orig, skel_list, slh, thr):

        _line_tracking.rescue_stafflines_using_mask( \
            self, orig, skel_list, slh, thr)

    __call__ = staticmethod(__call__)

#----------------------------------------------------------------

class rescue_stafflines_using_secondchord(PluginFunction):
    """Every point of a staff line is examined for chords
running through that point except the staff line, that are longer
than a certain threshold. For every chord found, the area it coveres
is \"rescued\" from the staff line removal.

- *original_image* is the image without the staff lines removed.

- *skeleton*: All skeletons in a nested list.

- *staffline_height*: (self-explaining)

- *staffspace_height*: (self-explaining)

- *threshold*: Threshold for chord detection: This represents
  the minimum chordlength.

- *peak_depth*: The minimum depth of a \"valley\" between two
  maximums in the histogram so that they will be treated as peaks.

- *direction*: when 0, the entire vertcial black run is rescued,
  when 1, the black run in the direction of the scondchord is rescued.
"""

    category = None
    self_type = ImageType(ONEBIT)
    return_type = None
    args = Args([ImageType([ONEBIT], 'original_image'), \
                 Class("skeleton_list"), \
                 Int("staffline_height"), \
                 Int("staffspace_height"), \
                 Int("threshold"), \
                 Int("max_gap_height"), \
                 Int("peak_depth"), \
                 Int("direction", default = 0)])
#                Int("debug", default = 0), \
#                Int("dy"), \
#                Int("dx")])

    author = "Florian Pose"
    progress_bar = "Rescuing parts of superimposed objects..."

    def __call__(self, orig, skel_list, slh, \
                 ssh, thr, gap, pd, direction=0):
        # , dbg, dy, dx):

        _line_tracking.rescue_stafflines_using_secondchord( \
            self, orig, skel_list, slh, \
            ssh, thr, gap, pd, direction)
        #, dbg, dy, dx)

    __call__ = staticmethod(__call__)

#----------------------------------------------------------------

class MusicStaves_linetracking(PluginFunction):
    """Creates a `MusicStaves_linetracking`__ object.

.. __: gamera.toolkits.musicstaves.musicstaves_linetracking.MusicStaves_linetracking.html

Leave the parameters *staffspace_height* and *staffline_height* zero
unless you have computed them somehow beforehand. When left zero,
they are computed automatically.
"""
    pure_python = 1
    category = "MusicStaves/Classes"
    self_type = ImageType([ONEBIT, GREYSCALE])
    args = Args([Int('staffline_height', default=0),\
                 Int('staffspace_height', default=0)])
    return_type = Class("musicstaves")

    def __call__(image, staffline_height = 0, staffspace_height = 0):
        
        from gamera.toolkits.musicstaves import musicstaves_linetracking
        
        return musicstaves_linetracking.MusicStaves_linetracking( \
                image, staffline_height, staffspace_height)

    __call__ = staticmethod(__call__)
    author = "Christoph Dalitz"

#----------------------------------------------------------------

class line_tracking_module(PluginModule):
    category = None
    cpp_headers = ["line_tracking.hpp"]
    functions = [get_staff_skeleton_list, \
                 extract_filled_horizontal_black_runs, \
                 thinning_v_to_skeleton_list, skeleton_list_to_image, \
                 follow_staffwobble, remove_line_around_skeletons, \
                 rescue_stafflines_using_mask, \
                 rescue_stafflines_using_secondchord, \
                 MusicStaves_linetracking]
    author = "Thomas Karsten, Christoph Dalitz and Florian Pose"

module = line_tracking_module()

#----------------------------------------------------------------
