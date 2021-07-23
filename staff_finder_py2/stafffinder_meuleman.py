# -*- mode: python; indent-tabs-mode: nil; tab-width: 4 -*-
# vim: set tabstop=4 shiftwidth=4 expandtab:

#----------------------------------------------------------------

#
# Copyright (C) 2021 Mathias Meuleman
#
# Based on the work of Christoph Dalitz, Thomas Karsten and Florian Pose
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

from gamera.core import *
from gamera.core import Point as CorePoint

from gamera.plugins.structural import least_squares_fit
from gamera.toolkits.musicstaves.plugins import *

from gamera.args import *
from gamera.args import Point as ArgsPoint

from gamera.toolkits.musicstaves.stafffinder \
     import StaffFinder as _virtual_StaffFinder

from gamera.toolkits.musicstaves.stafffinder import StafflineSkeleton

#----------------------------------------------------------------

class StaffFinder_meuleman(_virtual_StaffFinder):
    
    """Finds staff lines with the aid of long quasi black run extraction
and skeletonization.

Based on the work as presented in:

    Dalitz, C., Droettboom, M., Pranzas, B., & Fujinaga, I. (2008).
    A comparative study of staff removal algorithms. IEEE transactions
    on pattern analysis and machine intelligence, 30(5), 753-766.

Modifications were made to improve staff finding staffs of varying staffline
counts. The original method allowed to pass *num_lines=0*, but its results
were found to be unstable. Below the amended work flow of this method can be
found. The following modifications were made to improve its results:

  - Step 3a: \"Remove line segments without links\" has been removed. This
    step would remove all staffs that consist of a single staffline.

  - Step 5a: \"Remove staffs that have too few staff liens\" has been removed.
    If *num_lines* was not set, it was estimated as the most frequent staffline
    count. Since we want to allow varying staffline counts however, this step
    is removed.

  - Step 9: \"Rectify incorrect groups splits and joins\" has been added.
    This serves as a final sanity check, where the distance between consecutive
    stafflines is used to determine whether groups of stafflines were rightfully
    separated, or need to be joined together, and vice versa.

The method works as follows:

1) Extract long horizontal runs with an average blackness above
   a certain threshold. The resulting filaments are vertically thinned
   and among the resulting lines, those shorter than 2 * *staffspace_height*
   are removed.

2) Vertically link line segments that overlap horizontally and
   have a vertical distance about *staff_space_height* + *staff_line_height*.

3) Remove line segments without links and label the remaining segments
   with a staff and line label. Staffs are identified as connected components
   and line numbers are assinged based upon vertical link steps.

4) \"Melt\" overlapping segments of the same staff line to one segment.
   This is done by selecting the segment that is closer to a least square
   fit of the total line.

5) Remove staffs that have too few staff lines and remove additional
   staff lines on top of and below the staff.

   - If the number of lines in every staff is given:
     Get the shortest of the upper and lower line and
     remove it. Do this until the correct number
     of staff lines is reached.

   - If the number of lines is unknown: Remove every
     line in a staff that is much shorter than the
     maximum length in this staff.

6) Remove staffline groups that overlap with other staffline groups.
   From all overlapping groups the widest is kept.
   Then join adjacent groups, whose distance is smaller then
   2 * *staffspace_height*

7) [Remove groups that are narrower than half the widest group.]
   This step has been removed!

8) Interpolate between non-overlapping skeleton segments of
   the same staff line.

9) Rectify incorrect group splits and joins
    - If vertical distance between lines in a group is
      larger than 3 * staffspace_height, break up the group.

    - If vertical distance between two groups is smaller
      than 1.5 * staff_space_height, join the groups.

:Author: Mathias Meuleman
:Original authors: Christoph Dalitz, Thomas Karsten and Florian Pose
"""

    #------------------------------------------------------------

    def __init__(self, img, \
                 staffline_height = 0, \
                 staffspace_height = 0):
        
        _virtual_StaffFinder.__init__(self, img, \
                                      staffline_height, \
                                      staffspace_height)
        
        self.find_staves_args = Args([Int("num_lines", default = 0), \
                                      Int("window", default = 3), \
                                      Int("blackness", default = 60), \
                                      Int("tolerance", default = 25), \
                                      Check("align_edges", default = True), \
                                      Check("join_interrupted", default = True), \
                                      Int("debug", default = 0)])

    #------------------------------------------------------------
    def find_staves(self, num_lines=0, window=3, blackness = 60, \
                    tolerance = 25, align_edges=True, join_interrupted=True, debug = 0):
        """Method for finding the staff lines.

Signature:

  ``find_staves(num_lines=0, window=3, blackness=60, tolerance=25, align_edges=True, join_interrupted=True, debug=0)``

with

- *num_lines*:

  Number of lines per staff. If unknown, set to 0.

- *window* and *blackness*:

  Parameters for the extraction of long horizontal runs. Only pixels are
  extracted that have an average blackness greater than *blackness* within
  a window of the width *window* \* *staff_space_height*.

- *tolerance*:

  The tolerance that is used while connecting line segments that
  belong to the same staff. They may have a vertical distance of
  *staff_line_height* + *staff_space_height* with a deviation of
  *tolerance* in percent.

- *align_edges*:

  When `True`, staff line edges in a staff are aligned up to the
  left most and right most found staff line within this staff.

- *debug*:

  0 = Be quiet.
  1 = Show Progress messages.
  2 = print images with prefix 'dalitzdebug' to current directory

This method fills the *self.linelist* attribute for further
processing.
"""

        #--------------------------------------------------------
        #
        # Step 1: Get staff skeleton list
        #

        if debug > 0:
            print
            print "Getting staff skeletons..."

        # Get the skeleton list
        skeleton_list = self.image.get_staff_skeleton_list( \
            self.staffline_height, \
            window * self.staffspace_height, \
            blackness)

        too_short_skeletons = [ line for line in skeleton_list \
                                if len(line[1]) < self.staffspace_height * 2 ]


        if len(too_short_skeletons) > 0:
            
            if debug > 0:
                print "   %i skeletons are too short. Removing." \
                      % (len(too_short_skeletons))

            # Remove very short segments
            for s in too_short_skeletons:
                skeleton_list.remove(s)

        max_dist = 2 * self.staffspace_height
        for line in skeleton_list:
            points = sorted(line[1])
            mid = len(points) // 2
            median = (points[mid] + points[~mid]) / 2
            for p in [p for p in line[1] if abs(median - p) > max_dist]:
                line[1].remove(p)

        # Create graph
        segments = []
        for line in skeleton_list:
            n = StaffSegment()
            n.row_start = min(line[1][0], line[1][-1])
            n.col_start = line[0]
            n.row_end = max(line[1][0], line[1][-1])
            n.col_end = n.col_start + len(line[1]) - 1
            n.skeleton = line		
            segments.append(n)

        #--------------------------------------------------------
        #
        # Step 2: Create vertical connections
        #
        #  A connection is done between two segments that
        #  overlap horizontally if the vertical distance
        #  is staff_line_height + staff_space_height
        #  (+- tolerance) percent
        #

        if debug > 0:
            print "Creating vertical connections..."

        connections = []

        tol = float(tolerance) / 100.0
        min_dist = (self.staffline_height + self.staffspace_height) \
                   * (1.0 - tol)
        max_dist = (self.staffline_height + self.staffspace_height) \
                   * (1.0 + tol)

        for seg1 in segments:
            for seg2 in segments:

                # No self-connections, segments must overlap
                if seg1 != seg2 and seg1.overlaps(seg2):

                    # Calculate vertical distance in
                    # the middle of the overlapping parts
                    mid = (max(seg1.col_start, seg2.col_start) + \
                           min(seg1.col_end, seg2.col_end)) / 2
                    row1 = seg1.skeleton[1][mid - seg1.col_start]
                    row2 = seg2.skeleton[1][mid - seg2.col_start]
                    dist = row2 - row1

                    if dist >= min_dist and dist <= max_dist:

                        # seg2 belongs to a staff
                        # line below seg1
                        if seg1.down_links.count(seg2) == 0:
                            seg1.down_links.append(seg2)
                        if seg2.up_links.count(seg1) == 0:
                            seg2.up_links.append(seg1)

                        # Add connection for debugging
                        conn = StaffConn()
                        conn.col = mid
                        conn.row_start = min(row1, row2)
                        conn.row_end = max(row1, row2)
                        connections.append(conn)

                    elif dist <= -min_dist and dist >= -max_dist:

                        # seg2 belongs to a staff
                        # line on top of seg1
                        if (seg2.down_links.count(seg1) == 0):
                            seg2.down_links.append(seg1)
                        if (seg1.up_links.count(seg2) == 0):
                            seg1.up_links.append(seg2)

        if debug > 0:
            print "   %i connections created." % (len(connections))

        if debug > 1:
            draw_segments(self.image, segments, 'step_2_segments')
        #--------------------------------------------------------
        #
        # Step 3a: Remove Segments without links
        #

        # if debug > 0:
        #     print "Removing segments without links..."
        #
        # tmp = []
        # remove = []
        # for seg in segments:
        #     if len(seg.down_links) > 0 or len(seg.up_links) > 0:
        #         tmp.append(seg)
        #     else:
        #         remove.append(seg)
        #
        # if debug > 0:
        #     print "   %i segments do not contain links. Removing." % (len(segments) - len(tmp))
        # if debug > 1:
        #     label = 64
        #     remove_group = StaffGroup()
        #     remove_group.label = label
        #     for s in remove:
        #         s.label = label
        #         remove_group.extend(s)
        #     draw_segments(self.image, [remove_group], 'step_3_removed')
        #
        # segments = tmp
        # del tmp

        #--------------------------------------------------------
        #
        # Step 3b: Label CC's and line numbers
        #
        #  Do a breadth-first search on the segments
        #  and give a unique label to every segment that
        #  belongs to a certain group of connected segments.
        #  Increase the line number with every downward
        #  connection and decrease it with every upward
        #  connection.

        if debug > 0:
            print "Grouping segments to staffs..."

        label = -1
        groups = []

        for segment in segments:

            # Segment already labeled: Process next one
            if segment.label != None: continue

            seg = segment
            label = label + 1 # Increase label
            group = StaffGroup()

            # Label this segment
            seg.label = label
            seg.line = 0
            group.extend(seg)

            # Label neighbors
            neighbors = []
            for n in seg.up_links:
                if n.label == None:
                    n.label = label
                    # Up-link: Decrease line number
                    n.line = seg.line - 1
                    group.extend(n)
                    neighbors.append(n)

                elif n.label != label:
                    raise RuntimeError, "Labelling error!"

            for n in seg.down_links:
                if n.label == None:
                    n.label = label
                    # Down-link: Increase line number
                    n.line = seg.line + 1
                    group.extend(n)
                    neighbors.append(n)

                elif n.label != label:
                    raise RuntimeError, "Labelling error!"

            # Process neighbors
            while len(neighbors) > 0:

                new_neighbors = []

                for seg in neighbors:
                    for n in seg.up_links:
                        if n.label == None:
                            n.label = label
                            # Up-Link: Decrease line number
                            n.line = seg.line - 1
                            group.extend(n)
                            new_neighbors.append(n)

                        elif n.label != label:
                            raise RuntimeError, "Labelling error!"

                    for n in seg.down_links:
                        if n.label == None:
                            n.label = label
                            # Down-Link: Increase line numver
                            n.line = seg.line + 1
                            group.extend(n)
                            new_neighbors.append(n)

                        elif n.label != label:
                            raise RuntimeError, "Labelling error!"

                neighbors = new_neighbors

            groups.append(group)

        if debug > 0:
            print "   Found %i staffs." % (len(groups))
        if debug > 1:
            draw_group_segments(self.image, groups, 'step_3_segments')
            draw_groups(self.image, groups, 'step_3')

        #--------------------------------------------------------
        #
        # Step 4: Melt overlapping segments of a staff line
        #
        #  If two segments of the same staff line overlap,
        #  they have to be melted to one, so that the later
        #  interpolation can assume non-overlapping segments.
        #
        #  In the overlapped parts, the decision of which
        #  part to use is made by laying a least square fit over
        #  the non-overlapping parts. The overlapping part, that
        #  fits better to the resulting straight line is used
        #  to substitute the overlapping range.

        if debug > 0:
            print "Melting overlapping line segments..."

        melt_skeletons = []

        melt_count = 0
        for g in groups:
            for l in range(g.min_line, g.max_line + 1):

                melted = True
                while (melted):
                    melted = False

                    for seg1 in g.segments:
                        if seg1.line != l: continue

                        for seg2 in g.segments:
                            if seg2.line != l or seg1 == seg2: continue

                            if seg1.overlaps(seg2):
                                melt_skeletons.append(seg1.melt(seg2))
                                g.segments.remove(seg2)
                                melted = True
                                melt_count = melt_count + 1

                                # Jump out of the
                                # inner 'for'
                                break

                        # Jump out of the outer 'for'
                        if melted: break

        if debug > 0 and melt_count > 0:
            print "   %d segments melted." % (melt_count)

        if debug > 1:
            draw_group_segments(self.image, groups, 'step_4_segments')
            draw_groups(self.image, groups, 'step_4')
        #--------------------------------------------------------
        #
        # Step 5a: Removal of staffs with too few lines
        #
        #  when the number of lines is not given, it is
        #  estimated as the most frequent num_lines among the wide groups
        #

        # Get maximum staff line width of all staffs
        max_group_width = 0
        for g in groups:
            width = g.col_end - g.col_start + 1
            if width > max_group_width: max_group_width = width

        # Step removed, since this is very impactful when estimated_num_lines is wrong
        # Get maximum staff line width of all staffs
        """
        # estimate num_lines
        if num_lines > 0:
            estimated_num_lines = num_lines
        else:
            num_hist = {}
            for g in groups:
                if g.col_end - g.col_start + 1 > max_group_width / 2:
                    n = g.max_line - g.min_line + 1
                    if num_hist.has_key(n):
                        num_hist[n] += 1
                    else:
                        num_hist[n] = 1
            max_count = 0
            estimated_num_lines = 0
            for (n,c) in num_hist.iteritems():
                if c > max_count:
                    estimated_num_lines = n
                    max_count = c
            print "num_lines estimated as ", estimated_num_lines

        # remove staffs with fewer lines
        if debug > 0:
            print "Removing staffs with fewer lines than", estimated_num_lines, "..."
        rem_groups = [g for g in groups \
                      if g.max_line - g.min_line + 1 < estimated_num_lines]
        for g in rem_groups: groups.remove(g)
        if debug > 0 and len(rem_groups) > 0:
            print "   %i removed, %i staffs left." \
                  % (len(rem_groups), len(groups))
        """

        #--------------------------------------------------------
        #
        # Step 5b: Remove additional lines above and below staffs
        #
        #  If the number of staff lines in a staff is known,
        #  the top or bottom line (the narrower one) is removed
        #  until the correct number of lines is reached.
        #
        #  If it is not known, every line is removed, that is
        #  much narrower than the maximum line length in this
        #  staff.
        #

        if debug > 0:
            print "Removing additional staff lines..."

        lines_removed = 0

        # In every staff group
        for g in groups:

            lengths = []
            max_length = 0

            # Calculate range of every staff line
            for line in range(g.min_line, g.max_line + 1):

                length_sum = 0
                for s in [seg for seg in g.segments if seg.line == line]:
                    length_sum = length_sum + s.col_end - s.col_start + 1
                lengths.append((line, length_sum))

                if length_sum > max_length: max_length = length_sum

            # If the real number of staff lines is given...
            if num_lines > 0:

                # While there are to much lines...
                while g.max_line - g.min_line + 1 > num_lines:

                    if lengths[0][1] < lengths[-1][1]: # Upper line shortest
                        g.remove_line(lengths[0][0])
                        lengths.pop(0) # Remove first line info

                    else: # Bottom line shortest
                        g.remove_line(lengths[-1][0])
                        lengths.pop() # Remove last line info

                    lines_removed = lines_removed + 1

            else: # Real number of lines not given

                # Simply remove lines that are too short
                for line, length in lengths:
                    if (length < max_length * 0.8):
                        g.remove_line(line)
                        lines_removed = lines_removed + 1

            # TODO: Check if groups have been seperated!
            # Fixed in new step 9

        if debug > 0 and lines_removed > 0:
            print "   Removed %d lines." % (lines_removed)

        if debug > 1:
            draw_group_segments(self.image, groups, 'step_5_segments')
            draw_groups(self.image, groups, 'step_5')
        #--------------------------------------------------------
        #
        # Step 6a: Remove groups that overlap with wider groups
        #

        if debug > 0:
            print "Removing embedded staffs..."

        # sort groups line for line from left to right
        def _cmp_y(g,h):
            if g.row_start < h.row_start: return -1
            else: return 1
        groups.sort(_cmp_y)
        ngroups = len(groups)
        breakdist = 2 * self.staffspace_height
        for i in range(ngroups):
            # find items in same line
            candidates = []
            for j in range(i+1,ngroups):
                if groups[i].row_end - breakdist < groups[j].row_start:
                    break
                candidates.append([j,groups[j]])
            # pick the leftmost as next in order
            if len(candidates) > 0:
                minj = i; min_col_start = groups[i].col_start
                for c in candidates:
                    if c[1].col_start < min_col_start:
                        minj = c[0]; min_col_start = c[1].col_start
                if minj > i:
                    g = groups[i]
                    groups[i] = groups[minj]
                    groups[minj] = g

        #print "groups sorted:"
        #for g in groups:
        #    print "rows = ", g.row_start, "-", g.row_end, "col_start =",g.col_start, "lines =", g.max_line - g.min_line + 1

        # when consecutive groups overlap, keep only the widest
        rem_groups = []
        i = 0; j = 0
        while i < len(groups) and j < len(groups) - 1:
            j += 1
            g = groups[i]; h = groups[j]
            if g.col_end >= h.col_start and \
               ((h.row_start < g.row_start and g.row_start < h.row_end) or \
                (h.row_start < g.row_end and g.row_end < h.row_end)):
                if (g.col_end - g.col_start) < (h.col_end - h.col_start):
                    rem_groups.append(g)
                    i = j
                else:
                    rem_groups.append(h)
            else:
                i += 1
        for g in rem_groups:
            if g in groups:
                groups.remove(g)
        if debug > 0 and len(rem_groups) > 0:
            print "   %i removed, %i staffs left." \
                  % (len(rem_groups), len(groups))

        if debug > 1:
            draw_group_segments(self.image, groups, 'step_6a_segments')
            draw_groups(self.image, groups, 'step_6a')
        #--------------------------------------------------------
        #
        # Step 6b: Join groups belonging to the same staff system
        #          (only executed when join_interrupted is set)
        #

        if join_interrupted:
            if debug > 0:
                print "Join interrupted staves..."
            # check whether consecutive groups follow each other
            # and how they could be linked
            # condition: distance < 2*staffspace_height
            rem_groups = []
            for i, g1 in enumerate(groups):

                if g1 in rem_groups:
                    continue

                for j in range(i + 1, len(groups)):
                    g2 = groups[j]

                    # join only if vertically overlapping
                    if max(g1.row_start, g2.row_start) > min(g1.row_end, g2.row_end):
                        break

                    # join groups with the same line count only
                    if g2.max_line - g2.min_line != g1.max_line - g1.min_line:
                        break

                    if g2.col_start <= g1.col_end:
                        break

                    if g2.col_start - g1.col_end >= 2*self.staffspace_height:
                        break

                    # now do the join thing
                    g1.join(g2)
                    rem_groups.append(g2)

            for g in rem_groups: groups.remove(g)

            if debug > 0:
                print "   %i group(s) joined." % len(rem_groups)

            if debug > 1:
                draw_group_segments(self.image, groups, 'step_6b_segments')
                draw_groups(self.image, groups, 'step_6b')

        #--------------------------------------------------------
        #
        # Step 7: Removal of narrow staffs
        #

        if debug > 0:
           print "Removing invalid staffs..."
        rem_groups = [g for g in groups \
                     if g.col_end - g.col_start + 1 < max_group_width / 2]

        for g in rem_groups: groups.remove(g)
        if debug > 0 and len(rem_groups) > 0:
           print "   %i removed, %i staffs left." \
                 % (len(rem_groups), len(groups))

        if debug > 1:
            draw_group_segments(self.image, groups, 'step_7_segments')
            draw_groups(self.image, groups, 'step_7')

        #--------------------------------------------------------
        #
        # Step 8: Interpolate broken staff lines
        #
        #  If there is more than one segment left for a staff
        #  line: Order and connect them.
        #

        if debug > 0:
            print "Connecting broken lines..."

        conn_skels = []
        conn_count = 0

        for g in groups:
            for line in range(g.min_line, g.max_line + 1):

                # Count segments in this line
                line_segs = []
                for s in g.segments:
                    if s.line == line: line_segs.append(s)

                # If more than one segment per line: Connect them!
                if len(line_segs) > 1:

                    conn_count = conn_count + len(line_segs)

                    # Sort segments by start column
                    line_segs.sort(lambda x, y: int(x.col_start - y.col_start))

                    s1 = line_segs.pop(0) # Leftmost segment

                    for s in line_segs:
                        conn_skel = s1.connect(s)
                        if conn_skel: conn_skels.append(conn_skel)
                        g.segments.remove(s)

        if debug > 0 and conn_count > 0:
            print "   %i connected" % (conn_count)

        if debug > 1:
            draw_group_segments(self.image, groups, 'step_8_segments')
            draw_groups(self.image, groups, 'step_8')


        #-----------------------------------------------------
        #
        # Step 9: Rectify incorrect group splits and joins
        #
        #  If vertical distance between lines in a group is
        #  larger than 3 * staffspace_height, break up the group.
        #  If vertical distance between two groups is smaller
        #  than 1.5 * staff_space_height, join the groups.
        #

        if debug > 0:
            print "Splitting up combined staffs..."

        split_group_count = 0
        tmp = []
        max_seg_dist = 3 * self.staffspace_height
        for g in groups:
            if len(g.segments) <= 0:
                continue
            current_group = StaffGroup()
            current_group.extend(g.segments[0])
            prev_row_end = g.segments[0].row_end

            # Sort segments by start column
            g.segments.sort(lambda x, y: int(x.row_start - y.row_start))

            # Check each gap between the segments in order
            for i in range(1, len(g.segments)):
                seg = g.segments[i]
                dist = seg.row_start - prev_row_end
                if dist > max_seg_dist:
                    tmp.append(current_group)
                    current_group = StaffGroup()
                    split_group_count += 1
                current_group.extend(seg)
                prev_row_end = seg.row_end
            tmp.append(current_group)

        groups = tmp
        groups.sort(_cmp_y)
        del tmp

        if debug > 0:
            print "    %i groups split, %i groups remain" % (split_group_count, len(groups))

        if debug > 1:
            draw_group_segments(self.image, groups, 'step_9a_segments')
            draw_groups(self.image, groups, 'step_9a')

        if debug > 0:
            print "Joining broken staffs..."

        join_group_count = 0
        max_dist = 1.5 * self.staffspace_height
        i = 0
        while i < len(groups) - 1:
            g, g_next = groups[i], groups[i + 1]
            if g_next.row_start - g.row_end <= max_dist:
                g.join(g_next)
                groups.remove(g_next)
                join_group_count += 1
            else:
                i += 1

        groups.sort(_cmp_y)

        if debug > 0:
            print '    %i groups joined, %i groups remain' % (join_group_count, len(groups))

        if debug > 1:
            draw_group_segments(self.image, groups, 'step_9b_segments')
            draw_groups(self.image, groups, 'step_9b')


        #--------------------------------------------------------
        #
        #  Visualization
        #

        if (debug > 1):

            rgb = Image(self.image, RGB)

            print
            print "Drawing group backgrounds..."
            for g in groups:
                color = RGBPixel(150 + (31 * g.label) % 106, \
                                 150 + (111 * (g.label + 1)) % 106, \
                                 150 + (201 * (g.label + 2)) % 106)

                rgb.draw_filled_rect((g.col_start, g.row_start), \
                                     (g.col_end, g.row_end), \
                                     color)

            print "Drawing original image..."
            rgb.highlight(self.image, RGBPixel(0, 0, 0))

            print "Highlighting staff line candidates..."
            staff_skeleton = self.image.skeleton_list_to_image(skeleton_list)
            rgb.highlight(staff_skeleton, RGBPixel(255, 150, 0)) # orange
            staff_skeleton_short = self.image.skeleton_list_to_image(\
                too_short_skeletons)
            rgb.highlight(staff_skeleton_short, RGBPixel(255, 0, 150)) # orange

            black_runs = self.image.extract_filled_horizontal_black_runs(\
                windowwidth = self.staffspace_height * window, \
                blackness = blackness)

            black_runs = black_runs.to_rgb()

            black_runs.highlight(staff_skeleton, RGBPixel(0, 255, 0))
            black_runs.save_PNG("dalitzdebug_blackruns.png")

            print "Highlighting group segments..."
            group_skeletons = []
            melted_skeletons = []
            conn_skeletons = []

            for g in groups:
                for seg in g.segments:
                    group_skeletons.append(seg.skeleton)

            group_image = self.image.skeleton_list_to_image(group_skeletons)
            rgb.highlight(group_image, RGBPixel(0, 255, 0)) # green

            # Errors out for some reason
            # print "Highlighting melted sections..."
            # melt_image = self.image.skeleton_list_to_image(melt_skeletons)
            # rgb.highlight(melt_image, RGBPixel(0, 255, 255)) # cyan

            print "Highlighting connections..."
            conn_image = self.image.skeleton_list_to_image(conn_skels)
            rgb.highlight(conn_image, RGBPixel(255, 255, 0)) # yellow

            print "Drawing segment markers..."
            for g in groups:
                for seg in g.segments:

                    color = RGBPixel(100 + ((71 * seg.line) % 156), 0, 0)

                    rgb.draw_marker((seg.col_start, seg.row_start), \
                                    self.staffline_height * 2, \
                                    3, color)

                    rgb.draw_marker((seg.col_end, seg.row_end), \
                                    self.staffline_height * 2, \
                                    3, color)

            print "Drawing links..."

            # All connections

            for c in connections:
                    rgb.draw_line((c.col, c.row_start), (c.col, c.row_end), \
                                  RGBPixel(0, 0, 255)) # blue

            # Connections of group segments

            for g in groups:
                for seg in g.segments:
                    for link in seg.down_links:
                        mid = (max(seg.col_start, link.col_start) + \
                               min(seg.col_end, link.col_end)) / 2
                        row1 = seg.skeleton[1][mid - seg.col_start]
                        row2 = link.skeleton[1][mid - link.col_start]
                        rgb.draw_line((mid, row1), (mid, row2), \
                                      RGBPixel(255, 0, 200)) # pink

            print "Writing file..."
            rgb.save_PNG("dalitzdebug_out.png")

        #--------------------------------------------------------
        #
        # Copy over the results into self.linelist
        #

        self.linelist = []

        for g in groups:
            newstaff = []
            #for line in range(g.min_line, g.max_line + 1):
            for s in g.segments:
                skel = StafflineSkeleton()
                skel.left_x = s.skeleton[0]
                skel.y_list = s.skeleton[1]
                newstaff.append(skel)
            # sort by y-position
            newstaff.sort(lambda s1,s2: int(s1.y_list[0] - s2.y_list[0]))    
            self.linelist.append(newstaff)

        #--------------------------------------------------------
        #
        # Adjust edge points to the left/right most point with each staff
        #

        if align_edges:
            if debug > 0:
                print "Align edge points"
            for staff in self.linelist:
                # find left/right most edge point
                lefti = 0; left = self.image.ncols
                righti = 0; right = 0
                for i,skel in enumerate(staff):
                    if skel.left_x < left:
                        lefti = i; left = skel.left_x
                    if skel.left_x + len(skel.y_list) - 1 > right:
                        righti = i; right = skel.left_x + len(skel.y_list) - 1
                leftref = staff[lefti].y_list
                rightref = staff[righti].y_list
                # extrapolate left edge points
                for skel in staff:
                    if skel.left_x > left:
                        if skel.left_x - left < len(leftref):
                            dy = skel.y_list[0] - leftref[skel.left_x - left]
                        else:
                            dy = self.staffspace_height
                        x = skel.left_x - 1
                        while (x >= left):
                            if x-left < len(leftref):
                                skel.y_list.insert(0, leftref[x-left] + dy)
                            else:
                                skel.y_list.insert(0, skel.y_list[0])
                            x -= 1
                        skel.left_x = left
                # extrapolate right edge points
                for skel in staff:
                    if skel.left_x + len(skel.y_list) - 1 < right:
                        dy = skel.y_list[-1] - rightref[len(skel.y_list)]
                        x = skel.left_x + len(skel.y_list)
                        while (x <= right):
                            skel.y_list.append(rightref[x-left] + dy)
                            x += 1

        if debug > 0:
            print "Ready."
            print

#----------------------------------------------------------------

def draw_segments(original, segments, name):
    rgb = Image(original, RGB)
    color = RGBPixel(255, 125, 0)

    for seg in segments:
        # color = RGBPixel(150 + (31 * seg.label) % 106,
        #                  150 + (111 * (seg.label + 1)) % 106,
        #                  150 + (201 * (seg.label + 2)) % 106)
        rgb.draw_filled_rect((seg.col_start, seg.row_start), (seg.col_end, seg.row_end), color)
    rgb.highlight(original, RGBPixel(0, 0, 0))
    rgb.save_PNG('%s.png' % name)

def draw_group_segments(original, groups, name):
    rgb = Image(original, RGB)
    for g in groups:
        color = RGBPixel(150 + (31 * g.label) % 106,
                         150 + (111 * (g.label + 1)) % 106,
                         150 + (201 * (g.label + 2)) % 106)

        for seg in g.segments:
            rgb.draw_filled_rect((seg.col_start-5, seg.row_start-5), (seg.col_end+5, seg.row_end+5), color)
    rgb.highlight(original, RGBPixel(0, 0, 0))
    rgb.save_PNG('%s.png' % name)

def draw_groups(original, groups, name):
    rgb = Image(original, RGB)

    for g in groups:
        color = RGBPixel(150 + (31 * g.label) % 106,
                         150 + (111 * (g.label + 1)) % 106,
                         150 + (201 * (g.label + 2)) % 106)

        row_start, row_end = g.row_start, g.row_end
        if row_end - row_start < 20:
            row_start = row_start - 10
            row_end = row_end + 10
        rgb.draw_filled_rect((g.col_start, row_start), (g.col_end, row_end), color)
    rgb.highlight(original, RGBPixel(0, 0, 0))
    rgb.save_PNG('%s.png' % name)


class StaffSegment:

    """Class to describe a segment of a staff line."""

    #------------------------------------------------------------
    
    def __init__(self):

        """Constructs a segment."""
        
        self.row_start = 0
        self.col_start = 0
        self.row_end = 0
        self.col_end = 0
        self.skeleton = None
        self.up_links = []
        self.down_links = []
        self.label = None
        self.line = None        

    #------------------------------------------------------------
	
    def overlaps(self, other):

        """Returns 'True', if the current segments horizontally
        overlaps with the given one."""
	
        return (self.col_end >= other.col_start \
                and self.col_start <= other.col_end)

    #------------------------------------------------------------

    def melt(self, other):

        """Melts to overlapping segments together using the
        overlapping part of the segment that fits better into
        the environmental line. Returns a skeleton of the
        melted part."""

        # Remove all links to the segment to melt
        # Add links to the current segment instead
	
        for up_link in other.up_links:
            up_link.down_links.remove(other)
            if up_link.down_links.count(self) == 0:
                up_link.down_links.append(self)
            if self.up_links.count(up_link) == 0:
                self.up_links.append(up_link)
                    
        other.up_links = []
		
        for down_link in other.down_links:
            down_link.up_links.remove(other)
            if down_link.up_links.count(self) == 0:
                down_link.up_links.append(self)
            if self.down_links.count(down_link) == 0:
                self.down_links.append(down_link)
                    
        other.down_links = []

        # Calculate the least squares fit of the non-overlapping
        # parts of the segments

        points = []
        col_overlap = -1
        len_overlap = 0
        for col in range(min(self.col_start, other.col_start), \
                         max(self.col_end, other.col_end) + 1):

            in_self = col >= self.col_start and col <= self.col_end
            in_other = col >= other.col_start and col <= other.col_end

            if in_self and not in_other:
                points.append( \
                    CorePoint(col, \
                              self.skeleton[1][col - self.col_start]))
                
            elif in_other and not in_self:
                points.append( \
                    CorePoint(col, \
                              other.skeleton[1][col - other.col_start]))
                
            elif in_self and in_other:
                if col_overlap == -1: col_overlap = col
                len_overlap = len_overlap + 1
                
            else:
                raise RuntimeError, \
                      "Trying to melt non-overlapping segments!"

        if len(points) < 2:
            return

        # Do the least square fit
        (m, b, q) = least_squares_fit(points)

        # Calculate cumulative errors of overlapping parts

        self_error = 0
        other_error = 0
        for col in range(col_overlap, col_overlap + len_overlap):
            lsf_row = long(m * col + b + 0.5)
            self_row = self.skeleton[1][col - self.col_start]
            other_row = other.skeleton[1][col - other.col_start]
            self_error = self_error + abs(self_row - lsf_row)
            other_error = other_error + abs(other_row - lsf_row)

        skel = []
        melt = []
        for col in range(min(self.col_start, other.col_start), \
                         max(self.col_end, other.col_end) + 1):

            in_self = col >= self.col_start and col <= self.col_end
            in_other = col >= other.col_start and col <= other.col_end

            if in_self and not in_other:
                skel.append(self.skeleton[1][col - self.col_start])
                                       
            elif not in_self and in_other:
                skel.append(other.skeleton[1][col - other.col_start])

            elif in_other and in_self:
                 if (self_error <= other_error):
                     skel.append(self.skeleton[1][col - self.col_start])
                     melt.append(long(self.skeleton[1][col - self.col_start]))
                 else:
                     skel.append(other.skeleton[1][col - other.col_start])
                     melt.append(long(other.skeleton[1][col - other.col_start]))
                                       
            else:
                raise RuntimeError, \
                      "Trying to melt non-overlapping segments!"
			
        del self.skeleton
        self.skeleton = [min(self.col_start, other.col_start), skel]

        self.row_start = min(self.skeleton[1][0], self.skeleton[1][-1])
        self.row_end = max(self.skeleton[1][0], self.skeleton[1][-1])
        self.col_start = self.skeleton[0]
        self.col_end = self.col_start + len(self.skeleton[1]) - 1
	    
        return [long(col_overlap), melt]

    #------------------------------------------------------------

    def connect(self, other):

        """Connects to non-overlapping segments. Returns a
        skeleton of the connection."""

        if self.col_end >= other.col_start:
            raise RuntimeError, \
                  "Trying to connect overlapping segments!"

        conn_skel = None

        if self.col_end < other.col_start - 1:

            # Calculate the inclination
            row_diff = other.row_start - self.row_end
            col_diff = other.col_start - self.col_end
            m = float(row_diff) / float(col_diff)

            conn_skel = []
            conn_skel.append(self.col_end + 1)
            conn_skel.append([])

            # Interpolate between the segments

            for col in range(self.col_end + 1, other.col_start):
                row = long(m * (col - self.col_end) + self.row_end + 0.5)
                self.skeleton[1].append(row)
                conn_skel[1].append(row)

        # Add the other segment
        self.skeleton[1].extend(other.skeleton[1])

        # Calculate new dimensions
        self.row_end = self.skeleton[1][-1]
        self.col_end = self.col_start + len(self.skeleton[1]) - 1

        # Return a skeleton of the connection (for debugging)
        return conn_skel

#----------------------------------------------------------------

class StaffGroup:

    """Describes a staff group with all of its lines
    and segments."""

    #------------------------------------------------------------
    
    def __init__(self):

        """Constructs a new staff group."""
        
        self.label = None
        self.row_start = 0
        self.col_start = 0
        self.row_end = 0
        self.col_end = 0
        self.min_line = 0
        self.max_line = 0
        self.segments = []

    #------------------------------------------------------------

    def join(self, group2):

        """Add the segments of group2 to this group, adjusting
        their line attribute."""
        
        line_ofs = self.min_line - group2.min_line

        for s in group2.segments:
            s.line += line_ofs
            self.segments.append(s)

        self.update()
            

    #------------------------------------------------------------

    def extend(self, segment):

        """Adds a new segment to the group and calculates the
        new dimensions of the group."""

        if self.label == None:
            # The new segment is the first segment of the group
            self.label = segment.label
	    
        elif segment.label != self.label:
            raise RuntimeError, "Illegal label of added segment!"

        self.segments.append(segment)

        # Calculate new dimensions
        self.update()

    #------------------------------------------------------------

    def remove_line(self, line):

        """Removes all segments of the group, that belong
        to a certain staff line."""

        if line < self.min_line or line > self.max_line:
            raise RuntimeError, "Line out of range!"

        remove_segments = []

        for s in [seg for seg in self.segments if seg.line == line]:

            # Remove all links _from_ other segments
                
            for up_link in s.up_links:
                up_link.down_links.remove(s)

            for down_link in s.down_links:
                down_link.up_links.remove(s)

            # Remove links _to_ other segments
            s.up_links = []
            s.down_links = []

            # Mark segment for removing
            remove_segments.append(s)

        # Remove the segments
        for s in remove_segments: self.segments.remove(s)

        # Calculate new group dimensions
        self.update()

    #------------------------------------------------------------

    def update(self):

        """Calculates the current group dimensions."""

        if len(self.segments) > 0:

            s = self.segments[0]
            self.row_start = min(s.row_start, s.row_end)
            self.col_start = min(s.col_start, s.col_end)
            self.row_end = max(s.row_start, s.row_end)
            self.col_end = max(s.col_start, s.col_end)
            self.min_line = s.line
            self.max_line = s.line

            for s in self.segments[1:]:

                self.row_start = min(self.row_start, \
                                     s.row_start, \
                                     s.row_end)
                self.col_start = min(self.col_start, \
                                     s.col_start, \
                                     s.col_end)
                self.row_end = max(self.row_end, \
                                   s.row_start, \
                                   s.row_end)
                self.col_end = max(self.col_end, \
                                   s.col_start, \
                                   s.col_end)
                self.min_line = min(self.min_line, s.line)
                self.max_line = max(self.max_line, s.line)

#----------------------------------------------------------------

class StaffConn:

    """Describes a staff connection (only for debugging purposes)."""

    #------------------------------------------------------------
    
    def __init__(self):

        """Constructs a new connection."""
        
        self.col = 0
        self.row_start = 0
        self.row_end = 0

#----------------------------------------------------------------
