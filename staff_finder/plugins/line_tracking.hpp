/*
 * Copyright (C) 2005 Thomas Karsten, Christoph Dalitz and Florian Pose
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _LINE_TRACKING_HPP_
#define _LINE_TRACKING_HPP_

using namespace Gamera;
using namespace std;

#define _USE_MATH_DEFINES // for Visual Studio
#include <math.h>
#include <queue>
#include <deque>
#include <vector>

#include <gamera.hpp>
#include <plugins/segmentation.hpp>

#define comp_middle(x, y) x+(y-x)/2
#define DEBUG(x) do {std::cerr << x;} while (0);

/***************************************************************/

template<class T>
OneBitImageView* extract_filled_horizontal_black_runs(T&, int, size_t);

template<class T>
PyObject* thinning_v_to_skeleton_list(T&, int);

template<class T>
typename ImageFactory<T>::view_type* skeleton_list_to_image(T&, PyObject*);

/***************************************************************/

template<class T>
static inline size_t down(T& image, size_t col, size_t row,
                          size_t middle_prev,
                          bool* has_neighbor, size_t* neighbor);

template<class T>
static inline size_t up(T& image, size_t col, size_t row,
                        size_t middle_prev,
                        bool* has_neighbor, size_t* neighbor);

static inline size_t get_middle(size_t, size_t, size_t, bool*, bool*);

template<class T>
static inline void label_pixels(T& image, size_t top, size_t middle,
                                size_t bottom,
                                size_t col, typename T::value_type label);

static inline void interpolate(PyObject* y_values, size_t start_pos);

template<class T>
static inline int slither_midpoint(T& image, int col, int row, int staffline_height);

template<class T>
void remove_stave(T&, PyObject *, int, int);

// private members
static int staffline_height = 0;

/****************************************************************
 * chord_length
 *
 * Returns the length of a black chord starting at the
 * given row and column with the given angle. Stops the
 * tracing process if a length of *limit* is reached.
 *
 * Parameters:      x, y: Starting point
 *             angle_rad: Chord angle in radian
 *                   tan: Pre-calculated tangent of
 *                        angle_rad
 *                 limit: Upper bound of the chord
 *                        length to follow
 *
 * Returns: Chord length, or *limit*, if chord is longer
 *
 * Florian Pose, 2005-09-13
 ***************************************************************/

template <class T>
unsigned int chord_length(const T &image, int x, int y,
                          double angle_rad, double tan,
                          unsigned int limit)
{
  int dx, dy, nrows, ncols, dir;
  unsigned int length2, limit2;
  typename T::value_type black_value;

  nrows = image.nrows();
  ncols = image.ncols();
  black_value = black(image);

  limit2 = limit * limit;
  dy = 0;
  dx = 0;
  length2 = 0;
  dir = (angle_rad >= -M_PI_4 && angle_rad <= 3.0 * M_PI_4 ? 1 : -1);

  while (x + dx < ncols && x + dx >= 0
         && y - dy < nrows && y - dy >= 0
         && image.get(Point(x + dx, y - dy)) == black_value
         && length2 <= limit2)
  {
    length2 = dx * dx + dy * dy;

    if (fabs(tan) < 1.0) {
      dx += dir;
      dy = int(dx * tan + 0.5);
    }
    else {
      dy += dir;
      dx = int(dy / tan + 0.5);
    }
  }

  return (unsigned int) sqrt((double) length2);
}

/*****************************************************************************
 * thinning_v_to_skeleton_list
 *
 * thinning of elements, so that (*always*) the vertical middle is found. the
 * return value is a skeleton list (a nested list) and it contains following
 * values:
 * [[x0, [yx00, yx01, ..., yx0nx0-1]], [x1, [yx10, yx11, yx1nx1-1]], ...,
 * [xn-1, [yxn-10, yxn-11, ..., yxn-1nxn-1-1]]]
 *
 * note: the image is structured into elements that are labeled (so after
 *       applying this function the pixel values are never 'black value (1)',
 *       but greater. comparing each pixel with 1 (value for black) will fail
 *       (use Gamera's 'is_black'-function instead).
 *
 * toom, 2005-02-21
 ****************************************************************************/

template<class T>
PyObject* thinning_v_to_skeleton_list(T& image, int sl_height)
{
	typename T::value_type black_value=black(image);
	typename T::value_type max_label=
		std::numeric_limits<typename T::value_type>::max();
	typename T::value_type label;

	// set the global staffline_height
	staffline_height=sl_height;

	size_t cur_col, cur_row;
	size_t top, middle, bottom;
	size_t neighbor = 0;
	bool has_neighbor;

	// indicate the state of how the middle was computed
	bool guessed, guessed_prev;
	bool wall;

	// marker for where to start when interpolating
	int start_pos;
	int size;

	PyObject* list;
	PyObject* line;
	PyObject* y_values;
	PyObject* py_x;
	PyObject* py_y;

	list=PyList_New(0);

	// pixel values greater 1 are used to mark
	// pixels that have already been scanned
	label=1;

	/*
	 * scan the image row by row, column by column
	 */
	for (size_t col=0; col < image.ncols(); col++) {
		for (size_t row=0; row < image.nrows(); row++) {

			// this pixel has been scanned before or is white
			if (image.get(Point(col, row)) != black_value)
				continue;

			// initialize a python list for this line
			line=PyList_New(2);
			py_x=PyLong_FromLong(col);
			y_values=PyList_New(0);
			PyList_SetItem(line, 0, py_x);
			PyList_SetItem(line, 1, y_values);

			start_pos=0;

			// set middle to 0 to mark
			// the beginning of a new line
			middle=0;
			guessed_prev=false;
			wall=false;

			cur_col=col;
			cur_row=row;

			label++;
			if (label == max_label)
				throw std::range_error("Max label exceeded:"
						" Change OneBitPixel type in"
						" Gamera's pixel.hpp.");

			/*
			 * trace the found element from the left to the
			 * right to get the middle of a "perfect" staffline.
			 * the pixels of the possibly found staffline are
			 * marked (labeled).
			 */
			do {
				top=up(image, cur_col, cur_row, middle,
						&has_neighbor, &neighbor);
				bottom=down(image, cur_col, cur_row, middle,
						&has_neighbor, &neighbor);
				middle=get_middle(top, middle, bottom,
						&guessed, &wall);

				// add the middle to the python list
				py_y=PyLong_FromLong(middle);
				PyList_Append(y_values, py_y);
                Py_DECREF(py_y);

				/*
				 * special cases where the last values for
				 * the middle have to be adjusted
				 */
				size=PyList_GET_SIZE(y_values);
				if (wall) {
					if (size < 6*staffline_height+1)
						start_pos=0;
					else
						start_pos=size-1-6*
							staffline_height;

					interpolate(y_values, start_pos);

					wall=false;
					guessed=false;

				// middle has lost its border lines:
				// remember the starting pixel
				} else if (guessed && !guessed_prev)
					start_pos=size-1;

				// middle got back its border lines, so try
				// to correct the values from 'start_pos'
				else if (!guessed && guessed_prev)
					interpolate(y_values, start_pos);

				guessed_prev=guessed;

				label_pixels(image, top, middle, bottom,
						cur_col, label);

				cur_row=neighbor;
				cur_col++;
			} while (has_neighbor);

			PyList_Append(list, line);
            Py_DECREF(line);
		}
	}
	return list;
}

/*****************************************************************************
 * interpolate values of the list: adjust the values between 'start_pos' and
 * the last position in 'y_values'
 *
 * 2005-03-11
 ****************************************************************************/

static inline void interpolate(PyObject* y_values, size_t start_pos)
{
	PyObject* py_y;
	double x_diff, y_diff;
	double tan_a;
	int start_middle, cur_middle;
	int size;

	/*
	 * get the needed middle values for computation from the list
	 */
	size=PyList_GET_SIZE(y_values);
	py_y=PyList_GET_ITEM(y_values, start_pos);
	start_middle=PyLong_AsLong(py_y);
	py_y=PyList_GET_ITEM(y_values, size-1);
	cur_middle=PyLong_AsLong(py_y);

	/*
	 * compute the angle
	 */
	y_diff=(double)cur_middle-start_middle;
	x_diff=size-start_pos;
	tan_a=y_diff/x_diff;

	/*
	 * actual computation and resetting of the middle values
	 */
	for (size_t i=0; i < size-start_pos; i++) {
		py_y=PyLong_FromDouble(i*tan_a+start_middle);
		PyList_SET_ITEM(y_values, i+start_pos, py_y);
	}
}

/*****************************************************************************
 * mark the pixels above and below 'middle' (up to a certain value) to belong
 * to 'label'.
 *
 * 2005-03-10
 ****************************************************************************/

template<class T>
static inline void label_pixels(T& image,
                                size_t top, size_t middle, size_t bottom,
                                size_t col, typename T::value_type label)
{
	size_t decision_plus=middle+2*staffline_height;
	size_t decision_minus;
	size_t b, t;

	/*
	 * decide about until where to go
	 */
	if (middle < (size_t)2*staffline_height)
		decision_minus=0;
	else
		decision_minus=middle-2*staffline_height;

	if (bottom < decision_plus)
		b=bottom;
	else
		b=decision_plus;

	if (top > decision_minus)
		t=top;
	else
		t=decision_minus;

	/*
	 * finally label each pixel
	 */
	for (size_t i=t; i <= b; i++)
		image.set(Point(col, i), label);
}

/*****************************************************************************
 * check for black pixels below the current one (image[row][col]).
 * additionally, the column to the right is scanned for a black pixel that is
 * unlabeled yet and 'has_neighbor' is set to true, if a pixel was found (then
 * 'neighbor' stores its row)
 *
 * 2005-02-21
 ****************************************************************************/

template<class T>
static inline size_t down(T& image,
                          size_t col, size_t row,
                          size_t middle_prev,
                          bool* has_neighbor,
                          size_t* neighbor)
{
	typename T::value_type black_value=black(image);

	size_t bottom;
	bool last_col;
	double distance=(double)image.nrows();
	double nearest_distance=(double)image.nrows();

	bottom=row++;

	if (col+1 >= image.ncols())
		last_col=true;
	else
		last_col=false;

	if (last_col)
		while (row < image.nrows() && image.get(Point(col, row)) == black_value)
          bottom=row++;
	else if (*has_neighbor) {
		nearest_distance=fabs((double)middle_prev-*neighbor);

		while (row < image.nrows() && image.get(Point(col, row)) == black_value) {

			if (image.get(Point(col + 1, row)) == black_value) {
				distance=fabs((double)middle_prev-row);

				if (distance < nearest_distance) {
					nearest_distance=distance;
					*neighbor=row;
				}
			}
			bottom=row++;
		}
	} else
		while (row < image.nrows() && image.get(Point(col, row)) == black_value) {
			bottom=row;

			if (image.get(Point(col + 1, row)) == black_value) {
				if (*has_neighbor) {
					distance=
						fabs((double)middle_prev-row);

					if (distance < nearest_distance) {
						nearest_distance=distance;
						*neighbor=row;
					}
				} else {
					nearest_distance=
						fabs((double)middle_prev-row);
					*neighbor=row;
					*has_neighbor=true;
				}
			}
			row++;
		}

	return bottom;
}

/*****************************************************************************
 * check for black pixels above the current one (image[row][col]).
 * additionally, the column to the right is scanned for a black pixel that is
 * unlabeled yet and 'has_neighbor' is set to true, if a pixel was found (then
 * 'neighbor' stores its row). 'middle_prev' indicates the row of the previous
 * middle (or zero, when tracing a new line)
 *
 * 2005-02-21
 ****************************************************************************/

template<class T>
static inline size_t up(T& image,
                        size_t col, size_t row,
                        size_t middle_prev,
                        bool* has_neighbor,
                        size_t* neighbor)
{
	typename T::value_type black_value=black(image);

	size_t top;
	bool last_col;

	/*
	 * is this the last column of the image? if so, there cannot
	 * exist a neighbor pixel
	 */
	*has_neighbor=false;
	if (col+1 >= image.ncols())
		last_col=true;
	else {
		last_col=false;

		// try to find a neighbor in the same segment (previous
		// middle), this helps staying on the staffline
		if (middle_prev != 0)
			if (image.get(Point(col + 1, middle_prev)) == black_value) {
				*neighbor=middle_prev;
				*has_neighbor=true;
			}
	}

	/*
	 * go up this segment and stop when the top is found
	 */
	top=row;
	if (*has_neighbor || last_col) {
		while (image.get(Point(col, row)) == black_value) {
			top=row;
			if (row == 0)
				break;

			row--;
		}
	} else {
		*neighbor=0;
		double distance=(double)image.nrows();
		double nearest_distance=(double)image.nrows();

		/*
		 * additionally: test for a neighbor
		 */
		while (image.get(Point(col, row)) == black_value) {
			top=row;

			/*
			 * neighbor detected: try to get the neighbor
			 * that has the lowest distance to the previous
			 * middle
			 */
			if (image.get(Point(col + 1, row)) == black_value) {
				if (*has_neighbor) {
					distance=
						fabs((double)middle_prev-row);

					if (distance < nearest_distance) {
						nearest_distance=distance;
						*neighbor=row;
					}
				} else {
					*neighbor=row;
					nearest_distance=
						fabs((double)middle_prev-row);
					*has_neighbor=true;
				}
			}

			if (row == 0)
				break;

			row--;
		}
	}
	return top;
}

/*****************************************************************************
 * skeleton_list_to_image
 *
 * convert the skeleton list 'list' to an image.
 *
 * toom, 2005-02-23
 ****************************************************************************/

template<class T>
typename ImageFactory<T>::view_type* skeleton_list_to_image(T& image,
                                                            PyObject* list)
{
	typedef typename ImageFactory<T>::data_type data_t;
	typedef typename ImageFactory<T>::view_type view_t;

	data_t* data = new data_t(image.size(), Point(image.offset_x(), image.offset_y()));
	view_t* view = new view_t(*data);

	typename T::value_type black_value=black(*data);

	if (!PyList_Check(list))
		throw std::runtime_error("Must be a Python list.");

	int nrows=PyList_GET_SIZE(list);
	if (nrows < 0)
		throw std::runtime_error("List is too big.");

	PyObject* line;
	PyObject* y_values;
	PyObject* px_x;
	PyObject* px_y;

	size_t x;
	size_t y;

	for (int row=0; row < nrows; row++) {
		line=PyList_GET_ITEM(list, row);
		if (!PyList_Check(line) || (PyList_GET_SIZE(line) != 2))
			throw std::runtime_error("Must be a nested Python"
					" list.");

		px_x=PyList_GET_ITEM(line, 0);
		if (!PyLong_Check(px_x))
			throw std::runtime_error("List contains non-pixel"
					" values.");
		else
			x=(size_t)PyLong_AsLong(px_x);

		y_values=PyList_GET_ITEM(line, 1);
		if (!PyList_Check(y_values))
			throw std::runtime_error("Must be a nested Python"
					" list.");

		int ncols=PyList_GET_SIZE(y_values);
		if (ncols < 0)
			throw std::runtime_error("List is too big.");

		for (int col=0; col < ncols; col++) {
			px_y=PyList_GET_ITEM(y_values, col);
			if (!PyLong_Check(px_y))
				throw std::runtime_error("List contains"
						" non-pixel values.");
			else
				y=(size_t)PyLong_AsLong(px_y);

			if (x+col >= image.ncols() || y >= image.nrows())
				throw std::runtime_error("Values out of"
						" range.");
			else
				view->set(Point(x + col, y), black_value);
		}
	}
	return view;
}

/*****************************************************************************
 * return the "best" middle between 'top' and 'bottom'.
 *
 * 'middle_prev' is the returned middle of the previous computation (when set
 * to 0 it is ignored).
 * 'guessed' is set to false, when the middle could be computed. 'wall' is set
 * to true, when there is no continuous line between the previous and the
 * current middle.
 *
 * 2005-03-07 toom
 ****************************************************************************/

static inline size_t get_middle(size_t top,
                                size_t middle_prev,
                                size_t bottom,
                                bool* guessed,
                                bool* wall)
{
	// assume that the middle is not guessed but computed
	// and that the middle is not hitting the wall
	*guessed=false;
	*wall=false;

	/*
	 * return the real middle if it is the first middle value to return
	 */
	if (middle_prev == 0)
		return comp_middle(top, bottom);

	// tolerated distance to bottom or top
	size_t staff_height_0_5=(size_t)(staffline_height*0.75);

	/*
	 * try to continue tracing the previous middle (stay in the same
	 * segment) and try to keep contact to its bottom or top
	 */
	if (middle_prev >= top && middle_prev <= bottom)

		if (bottom-top < (size_t)(staffline_height*1.5))
			return comp_middle(top, bottom);

		// stay with the bottom
		else if (bottom <= middle_prev+staff_height_0_5)
			return bottom > staff_height_0_5 ?
				bottom-staff_height_0_5 : 0;

		// stay with the top
		else {
			size_t diff=middle_prev > staff_height_0_5 ?
				middle_prev-staff_height_0_5 : 0;

			if (top >= diff)
				return top+staff_height_0_5;

			// borders are far away, so we just keep our position,
			// hoping all the best
			else {
				*guessed=true;
				return middle_prev;
			}
		}
	/*
	 * the segment of the previous middle ended: try to locate it
	 */
	else {
		*wall=true;

		// segment is below the previous middle
		if (middle_prev < top) {
			if (bottom-top >= (size_t)(staffline_height*1.5))
				return top+staff_height_0_5;

		// segment is above the previous middle
		} else if (middle_prev > bottom)
			if (bottom-top >= (size_t)(staffline_height*1.5))
				return bottom-staff_height_0_5;
	}

	// no segment matched, so return the computed middle
	return comp_middle(top, bottom);
}

/*****************************************************************************
 * extract_filled_horizontal_black_runs
 *
 * keeps track of each staff line from its beginning to its end, so that its
 * path can * be followed. gaps in a staff line are filled, depending on the
 * values * 'width' and 'blackness' (actually, long horizontal runs are
 * tracked).
 *
 * parameters: width: width of the window, spanning 'width' columns
 *                    (height is 1)
 *
 *             blackness: threshold that decides about when a pixel is set to
 *                    black (computed over the whole window width)
 *
 * 2005-02-10 toom
 ****************************************************************************/

template<class T>
OneBitImageView* extract_filled_horizontal_black_runs(T& image, int width,
                                                      size_t blackness)
{
  OneBitImageData* dest = new OneBitImageData(image.size(),
                                              Point(image.offset_x(),
                                                    image.offset_y()));
  OneBitImageView* dest_view = new OneBitImageView(*dest);

	if ((width < 1) || ((size_t)width > image.ncols()))
		return dest_view;

	// image iterators
	typedef typename T::row_iterator rowIterator;
	rowIterator row;
	typename rowIterator::iterator col;
	typename rowIterator::iterator col_back;

	// view iterators
	typedef OneBitImageView::row_iterator rowViewIterator;
	rowViewIterator view_row;
	typename rowViewIterator::iterator view_col;
	typename rowViewIterator::iterator view_col_back;

	OneBitImageData::value_type onebit_white_value=white(*dest);
	OneBitImageData::value_type onebit_black_value=black(*dest);

	typename T::value_type white_value=white(image);
	typename T::value_type black_value=black(image);

	// window specific items
	int w_half;     // middle of the window
	typename T::col_iterator behind;   // pixel behind the window
	int w_sum;      // sum of the pixel values within the window
	double w_avg;   // average pixel value
	queue<int> window;

	// others
	int i;          // simple counter
	float threshold;

	if (black_value > white_value)
		threshold=blackness/100.;
	else
		threshold=(blackness/100.)*white_value;

	/*
	 * go through all rows of the image
	 */
	for (row=image.row_begin(), view_row=dest_view->row_begin();
			row != image.row_end(); row++, view_row++) {

		while (!window.empty())
			window.pop();

		/*
		 * fill the queue here: the middle of the queue marks the
		 * current pixel value, so at the beginning of a new row
		 * the pixel values in the right of the current pixel are
		 * in the outside of the image, the pixels to the left side
		 * are taken from the image
		 */
		w_half=width/2;

		w_sum=0;
		w_avg=0.;

		// pixels outside the image
		for (i=0; i <= w_half; i++) {
			window.push(white_value);
			w_sum+=window.back();
		}

		register int b_xpos=0;

		// pixels inside the image
		for (behind=row.begin(); i < width; i++, behind++, b_xpos++) {
			window.push(*behind);
			w_sum+=window.back();
		}

		/*
		 * the first pixel that can be marked black usually lies on
		 * a black path. do not forget to track back the black pixels
		 * that are in front of the one that 'switches' the
		 * threshold (otherwise the lines are too short, due to
		 * 'width' and 'blackness').
		 */
		bool first=true;

		for (col=row.begin(), view_col=view_row.begin();
				col != row.end(); col++, view_col++) {

			// push the next pixel (or white, if outside)
			if ((size_t)b_xpos++ < image.ncols()) {
				window.push(*behind);
				behind++;
			} else
				window.push(white_value);

			w_sum-=window.front();
			w_sum+=window.back();
			if (black_value > white_value)
				w_avg=(double)w_sum/width;
			else
				w_avg=white_value-(double)w_sum/width;

            window.pop();

			if (w_avg >= threshold) {
				*view_col=onebit_black_value;

				/*
				 * trace back black pixels that are in front
				 * of this one
				 */
				if (first) {
					first=false;
					col_back=col;
					view_col_back=view_col;

					if (col_back != row.begin()) {
						col_back--;
						view_col_back--;
					}

					while (1) {
						if (*col_back != black_value)
							break;

						*view_col_back=
							onebit_black_value;

						if (col_back == row.begin())
							break;

						col_back--;
						view_col_back--;
					}
				}
			// w_avg does not pass threshold
			} else {
				/*
				 * from here, mark all pixels black until a
				 * white one is detected (to keep the black
				 * line as long as it is in reality). this
				 * step is not done in one action, since the
				 * window has to be refilled properly.
				 */
				if (!first) {
					if (*col == black_value)
						*view_col=onebit_black_value;
					else {
						*view_col=onebit_white_value;
						first=true;
					}
				} else
					*view_col=onebit_white_value;
			}
		}
	}
	return dest_view;
}


/*****************************************************************************
 * follow_staffwobble
 *
 * follows a wobbling staff line and returns the adjusted skeleton
 *
 * chris, 2005-06-27
 ****************************************************************************/

template<class T>
PyObject* follow_staffwobble(T& image,
                             PyObject* skeleton,
                             int staffline_height,
                             int debug)
{
  PyObject* pyob;
  int left_x, right_x, x;
  int starty, startx, starty_right, startx_right, starty_left, startx_left;
  size_t ny;
  IntVector y_list;
  IntVector y_new;

  // read input arguments
  pyob = PyObject_GetAttrString(skeleton,"left_x");
  left_x = (int)PyLong_AsLong(pyob);
  Py_DECREF(pyob);
  pyob = PyObject_GetAttrString(skeleton,"y_list");
  if (!PyList_Check(pyob))
    throw std::runtime_error("follow_staffwobble_skeleton: y_list is no list");
  ny = (size_t)PyList_Size(pyob);
  for (size_t i=0; i<ny; i++) {
    y_list.push_back(PyLong_AsLong(PyList_GetItem(pyob,i)));
  }
  Py_DECREF(pyob);
  right_x = left_x + ny - 1;

  // find starting point for line tracking near skeleton middle
  x = left_x + ny / 2;
  starty_right = -1;
  startx_right = -1;
  while ((starty_right < 0) && (x < (int)image.ncols()) && (x < left_x + (int)ny)) {
    startx_right = x;
    starty_right = slither_midpoint(image, x, y_list[x-left_x], staffline_height);
    x++;
  }
  x = left_x + ny / 2;
  starty_left = -1;
  startx_left = -1;
  while ((starty_left < 0) && (x > 0) && (x > left_x)) {
    startx_left = x;
    starty_left = slither_midpoint(image, x, y_list[x-left_x], staffline_height);
    x--;
  }

  // pick start point closer to center
  if ((starty_left < 0)  && (starty_right < 0)) {
    starty = -1; startx = -1;
  }
  else if (starty_left < 0) {
    starty = starty_right; startx = startx_right;
  }
  else if (starty_right < 0) {
    starty = starty_left; startx = startx_left;
  }
  else {
    x = left_x + ny / 2;
    if (abs(startx_left - x) < abs(startx_right - x)) {
      starty = starty_left; startx = startx_left;
    } else {
      starty = starty_right; startx = startx_right;
    }
  }
  if (debug > 0)
    printf("Startpoint (x,y) = (%d,%d)\n", startx, starty);

  // no start point found: copy over input values
  if (starty < 0) {
    y_new = y_list;
  }
  // start point found:
  // follow wobble from startx to right and from startx to left
  else {
    int lasty, y;
    y_new.push_back(starty);
    x = startx + 1;
    lasty = starty;
    while ((x <= right_x) && (x < (int)image.ncols())) {
      y = slither_midpoint(image, x, lasty, staffline_height);
      if (y >= 0) lasty = y;
      y_new.push_back(lasty);
      x++;
    }
    x = startx - 1;
    lasty = starty;
    while ((x >= left_x) && (x >= 0)) {
      y = slither_midpoint(image, x, lasty, staffline_height);
      if (y >= 0) lasty = y;
      y_new.insert(y_new.begin(),lasty);
      x--;
    }
  }

  // create output arguments
  pyob = PyObject_GetAttrString(skeleton,"__class__");
  PyObject* newskel = PyInstanceMethod_New(pyob);
  Py_DECREF(pyob);
  PyObject_SetAttrString(newskel, "left_x", PyLong_FromLong(left_x));
  pyob = PyList_New(ny);
  for (size_t i=0; i<ny; i++) {
    PyList_SetItem(pyob, i, PyLong_FromLong(y_new[i]));
  }
  PyObject_SetAttrString(newskel, "y_list", pyob);

  return newskel;
}

/*****************************************************************************
 * slither_midpoint
 *
 * Returns the slither midpoint around (col, row).
 * When no slither is found, -1 is returned.
 *
 * chris, 2005-06-27
 ****************************************************************************/

template<class T>
static inline int slither_midpoint(T& image,
                                   int col, int row,
                                   int staffline_height)
{
  int top, bot, top1, bot1, top2, bot2, center;
  int maxlength, minlength;

  //maxlength = (int)(staffline_height * 1.5) + 1;
  maxlength = staffline_height + 2;
  //minlength = staffline_height - 2;
  minlength = 0;

  // pixel black: take runlength around pixel
  if (is_black(image.get(Point(col,row)))) {
    bot = row;
    while ((bot < (int)image.nrows()-1) && is_black(image.get(Point(col,bot+1))))
      bot++;
    top = row;
    while ((top > 0) && is_black(image.get(Point(col, top - 1))))
      top--;
  }
  // pixel white: pick one of the closest runlengths above and below
  else {
    // closest runlength above
    bot1 = row - 1;
    while ((bot1>0) && is_white(image.get(Point(col, bot1))) &&
           (row-bot1 < maxlength))
      bot1--;
    if (is_white(bot1) || (row-bot1 >= maxlength)) {
      bot1 = -1; top1 = -1; // no blackrun above
    } else {
      top1 = bot1;
      while ((top1>0) && is_black(image.get(Point(col, top1 - 1))))
        top1--;
    }
    // closest runlength below
    top2 = row + 1;
    while ((top2 < (int)image.nrows()-1) && is_white(image.get(Point(col, top2))) &&
           (top2-row < maxlength))
      top2++;
    if (is_white(top2) || (top2-row >= maxlength)) {
      bot2 = -1; top2 = -1; // no blackrun below
    } else {
      bot2 = top2;
      while ((bot2 < (int)image.nrows()-1) && is_black(image.get(Point(col, bot2 + 1))))
        bot2++;
    }
    // select appropriate blackrun (the one with center closer to row)
    if ((bot2 < 0) && (bot1 < 0)) {
      return -1;
    }
    else if (bot2 < 0) {
      top = top1; bot = bot1;
    }
    else if (bot1 < 0) {
      top = top2; bot = bot2;
    }
    //else if (bot2 - top1 < maxlength) {
    //  // join black run interrupted by white speckle
    //  top = top1; bot = bot2;
    //}
    else {
      // choose slither with center closer to input row
      // (round off towards row)
      if (row - (bot1+top1+1)/2 < (bot2+top2)/2 - row) {
        top = top1; bot = bot1;
      } else {
        top = top2; bot = bot2;
      }
    }
  }

  // return midpoint of blackrun
  if ((top <= bot) && ((bot-top) < maxlength) && ((bot-top) > minlength)) {
    center = (top + bot) / 2;
    // round towards row
    if (center < row)
      center = (top + bot + 1) / 2;
    // do not allow too strong deviations from previous point
    if (abs(center-row) < staffline_height)
      return center;
    else
      return -1;
  } else {
    return -1;
  }
}

/****************************************************************
 * convert_skeleton
 *
 * Convenience function for converting a skeleton from
 * a PyObject to a Skeleton object.
 *
 * Florian Pose, 2005-09-13
 ***************************************************************/

struct Skeleton
{
  int left_x;
  list<int> y_list;
};

void convert_skeleton(PyObject *skel_py, Skeleton &skel)
{
  PyObject *pyob;
  int len_y, x;

  // Get leftmost column from PyObject
  pyob = PyObject_GetAttrString(skel_py, "left_x");
  if (!pyob) throw std::runtime_error("Skeleton has no left_x!");
  skel.left_x = (size_t) PyLong_AsLong(pyob);
  Py_DECREF(pyob);

  // Get list of row values from PyObject
  pyob = PyObject_GetAttrString(skel_py, "y_list");

  if (!pyob) throw std::runtime_error("Skeleton has no y_list!");

  if (!PyList_Check(pyob))
  {
    throw std::runtime_error("Skeleton y_list is no list!");
  }

  // Push all row values in a list
  skel.y_list.clear();
  len_y = PyList_Size(pyob);
  for (x = 0; x < len_y; x++)
  {
    skel.y_list.push_back(PyLong_AsLong(PyList_GetItem(pyob, x)));
  }
  Py_DECREF(pyob);
}

/****************************************************************
 * convert_skeleton_list
 *
 * Convenience function for converting skeletons from
 * a PyObject containing a nested list with skeletons
 * to a std::list<Skeleton>.
 *
 * Florian Pose, 2005-09-13
 ***************************************************************/

void convert_skeleton_list(PyObject *skel_list_py,
                           list<Skeleton> &skel_list)
{
  PyObject *pyob;
  int i, len_i, j, len_j;
  Skeleton skel;

  skel_list.clear();

  if (!PyList_Check(skel_list_py))
  {
    throw std::runtime_error("Skeleton list param is no list!");
  }

  len_i = PyList_Size(skel_list_py);
  for (i = 0; i < len_i; i++)
  {
    pyob = PyList_GetItem(skel_list_py, i);

    if (!PyList_Check(pyob))
    {
      throw std::runtime_error("Skeleton list param is no nested list!");
    }

    len_j = PyList_Size(pyob);
    for (j = 0; j < len_j; j++)
    {
      convert_skeleton(PyList_GetItem(pyob, j), skel);
      skel_list.push_back(skel);
    }
  }
}

/****************************************************************
 * get_staffline_slice
 *
 * Returns a list of rows that belong to a vertical "slice"
 * of a staff line. Gaps are allowed, if the parameter
 * *max_gap_height* is greater zero. The maximum height of the
 * returned slice is limited by *threshold* in both upper
 * and lower direction. Thus, the maximum returned height ist
 * *threshold* * 2.
 *
 * Florian Pose, 2005-09-29
 ***************************************************************/

template<class T>
list<int> get_staffline_slice(T& image, int root_row, int root_col,
                              int threshold, int max_gap_height)
{
  typename T::value_type black_value;
  int gap, row;
  list<int> slice;

  black_value = black(image);

  // Determine black runlength above point
  gap = max_gap_height;
  row = root_row;
  while (root_row - row <= threshold && row >= 0)
  {
    if (image.get(Point(root_col, row)) == black_value)
    {
      slice.push_back(row);
      gap = max_gap_height; // Reset gap tolerance
    }
    else if (gap-- == 0) break;

    row--;
  }

  slice.reverse();

  // Determine black runlength below point
  gap = max_gap_height;
  row = root_row + 1;
  while (row - root_row <= threshold && row < (int) image.nrows())
  {
    if (image.get(Point(root_col, row)) == black_value)
    {
      slice.push_back(row);
      gap = max_gap_height; // Reset gap tolerance
    }
    else if (gap-- == 0) break;

    row++;
  }

  return slice;
}

/****************************************************************
 * remove_line_around_skeleton
 *
 * Removes black lines lying 'under' every skeleton line
 * specified. If the vertical extent of the staff line
 * exceeds a limit, only a height of *staffline_height*
 * is removed.
 *
 * Florian Pose, 2005-09-13
 ***************************************************************/

template<class T>
void remove_line_around_skeletons(T& image,
                                  PyObject *skel_list_py,
                                  int staffline_height,
                                  int threshold,
                                  int max_gap_height)
{
  typename T::value_type white_value;
  list<Skeleton> skel_list;
  list<Skeleton>::const_iterator skel_i;
  list<int>::const_iterator y_i;
  int start_row, end_row, x, y;
  list<int> slice;

  white_value = white(image);

  // Convert skeleton list
  convert_skeleton_list(skel_list_py, skel_list);

  // For every skeleton
  for (skel_i = skel_list.begin(); skel_i != skel_list.end(); skel_i++)
  {
    // For every point on the skeleton
    for (x = skel_i->left_x, y_i = skel_i->y_list.begin();
         y_i != skel_i->y_list.end();
         x++, y_i++)
    {
      // Get staffline "slice"
      slice = get_staffline_slice(image, *y_i, x,
                                  threshold, max_gap_height);

      if (!slice.size()) continue;

      if (slice.back() - slice.front() <= threshold)
      {
        // If runlength is within the threshold, remove everything
        start_row = slice.front();
        end_row = slice.back();
      }
      else
      {
        // Otherwise remove only staffline_height around skeleton
        start_row = *y_i - staffline_height / 2;
        end_row = *y_i + staffline_height / 2;
      }

      // Remove around skeleton
      for (y = start_row; y <= end_row; y++)
      {
        image.set(Point(x, y), white_value);
      }
    }
  }
}

/****************************************************************
 * rescue_stafflines_using_mask
 *
 * Lets a 'T'-shaped mask run above and below every staff
 * line skeleton. If the lower pixel of the mask and one of
 * the upper pixels is black, the vertical 'slice' of the
 * staff line is copied in the rescue image.
 *
 * Florian Pose, 2005-09-13
 ***************************************************************/

template<class T, class U>
void rescue_stafflines_using_mask(T& rescue_image,
                                  const U& original_image,
                                  PyObject *skel_list_py,
                                  int staffline_height,
                                  int threshold)
{
  typename T::value_type black_value, white_value;
  list<Skeleton> skel_list;
  list<Skeleton>::const_iterator skel_i;
  list<int>::const_iterator y_i;
  int x, ya, yb;

  // Determine black and white values
  black_value = black(rescue_image);
  white_value = white(rescue_image);

  // Convert skeleton list
  convert_skeleton_list(skel_list_py, skel_list);

  // For every skeleton
  for (skel_i = skel_list.begin(); skel_i != skel_list.end(); skel_i++)
  {
    // For every point on the skeleton
    for (x = skel_i->left_x, y_i = skel_i->y_list.begin();
         y_i != skel_i->y_list.end();
         x++, y_i++)
    {
      ya = *y_i - threshold;
      yb = *y_i + threshold;

      // Check for borders.
      // TODO: In border case the line slice is not rescued!
      if (ya < 0
          || x - 1 < 0
          || yb >= (int) original_image.nrows()
          || x + 1 >= (int) original_image.ncols()) continue;

      // Check for black pixels above and beyond the staff line
      if ((original_image.get(Point(x, ya)) == black_value
           && (original_image.get(Point(x - 1, ya - 1)) == black_value
               || original_image.get(Point(x, ya - 1)) == black_value
               || original_image.get(Point(x + 1, ya - 1)) == black_value))
          || (original_image.get(Point(x, yb)) == black_value
              && (original_image.get(Point(x - 1, yb + 1)) == black_value
                  || original_image.get(Point(x, yb + 1)) == black_value
                  || original_image.get(Point(x + 1, yb + 1)) == black_value)))
      {
        // Copy the staff line slice to the rescue image

        for (ya = *y_i;
             ya >= 0
               && original_image.get(Point(x, ya)) == black_value
               && *y_i - ya < staffline_height;
             ya--)
        {
          rescue_image.set(Point(x, ya), black_value);
        }

        for (yb = *y_i + 1;
             yb < (int) original_image.nrows()
               && original_image.get(Point(x, yb)) == black_value
               && yb - *y_i < staffline_height;
             yb++)
        {
          rescue_image.set(Point(x, yb), black_value);
        }
      }
    }
  }
}

/*** SECONDCHORD ALGORITHM *************************************/

//#define DEBUG_SECONDCHORD

/****************************************************************
 * ExtAngle
 *
 * Structure for an angle and its opposite plus precalculated
 * trigonometric functions for both.
 *
 * Florian Pose, 2005-09-29
 ***************************************************************/

typedef struct
{
  double angle_rad;
  double angle_deg;
  double tan;
  double sin;
  double opp_angle_rad;
  double opp_tan;
}
ExtAngle;

/****************************************************************
 * HistVal
 *
 * Structure of a histogram value with length and angle.
 *
 * Florian Pose, 2005-10-20
 ***************************************************************/

class HistVal
{
public:
  int length;
  int opp_length;
  int sum_length;
  unsigned int position;
  const ExtAngle *angle;
  HistVal() {length=0; opp_length=0; sum_length=0; position=0; angle=0;}
};


/****************************************************************
 * max_greater_than
 *
 * Predicate function on HistVal structure to use with
 * std::sort().
 *
 * Florian Pose, 2005-10-20
 ***************************************************************/

bool max_greater_than(const HistVal &left, const HistVal &right)
{
  return left.sum_length > right.sum_length;
}

/****************************************************************
 * has_secondchord1
 *
 * Analyzes a histogram and returns true, if the
 * following criteria match:
 *
 * - There is a chord height (not length!) of more than
 *   *side_level* in the side zone: fabs(angle_deg) > *side_zone*.
 *
 * or
 *
 * - At least one peak higher than *center_level* lies in
 *   the center zone: fabs(angle_deg) < *center_zone*.
 *
 *   and
 *
 *   the histogram has at least two peaks with a "valley"
 *   deeper than *depth* between them and that have a minimum
 *   distance of *min_peak_dist*.
 *
 * The index of the second chord in the passed HistVal-vector
 * is returned in histindex
 *
 * Florian Pose, 2005-10-06
 ***************************************************************/
bool has_secondchord1(const vector<HistVal> &hist,
                     int min_valley_depth,
                     double center_zone, int center_level,
                     double side_zone, int side_level,
                     double min_peak_dist, int* histindex
#ifdef DEBUG_SECONDCHORD
                     , int debug = 0
#endif
                     )
{
  vector<HistVal> peaks;
  HistVal last;
  bool inc, has_center_peak;
  unsigned int i, j, start, end;
  int limit;

  if (!hist.size()) throw std::runtime_error("Empty histogram!");
  *histindex = -1;

  // Run through every value of the histogram. Check
  // the chords in the side zone for a height of more
  // than *side_level*.

  for (i = 0; i < hist.size(); i++)
  {
    if (fabs(hist[i].angle->angle_deg) > side_zone
        && hist[i].sum_length * fabs(hist[i].angle->sin) >= side_level)
    {
#ifdef DEBUG_SECONDCHORD
      if (debug) printf("Has long side chord at %i!\n",
                        (int) hist[i].angle->angle_deg);
#endif
      return true;
    }
  }

  // Run through every value in the histogram
  // and create a list of the maxima

  last = hist.front();
  inc = true;

  for (i = 0; i < hist.size(); i++)
  {
    if (inc)
    {
      if (hist[i].sum_length < last.sum_length)
      {
        peaks.push_back(last);
        inc = false;
      }
    }
    else
    {
      if (hist[i].sum_length > last.sum_length)
      {
        inc = true;
      }
    }

    last = hist[i];
  }

  // If the end of the histogram remains ascending,
  // mark it as maximum, unless the first histogram
  // value is a maximum too (cyclic boundary).

  if (inc && (!peaks.size() || peaks.front().position == 0))
  {
    peaks.push_back(hist.back());
  }

  // Check if the histogram has a peak in the center zone
  // i.e. whether a staff line crosses the pixel

  has_center_peak = false;

  for (i = 0; i < peaks.size() - 1; i++)
  {
    if (fabs(peaks[i].angle->angle_deg) < center_zone
        && peaks[i].sum_length >= center_level)
    {
      has_center_peak = true;
      break;
    }
  }

  if (!has_center_peak)
  {
#ifdef DEBUG_SECONDCHORD
    if (debug) printf("Has no center peak!\n");
#endif
    return false;
  }

  // Reverse-sort maximum vector
  sort(peaks.begin(), peaks.end(), max_greater_than);

#ifdef DEBUG_SECONDCHORD
  if (debug)
  {
    printf("Peak list:\n");

    for (i = 0; i < peaks.size(); i++)
    {
      printf("%i) %i at %lf\n",
             i, peaks[i].sum_length, peaks[i].angle->angle_deg);
    }
  }
#endif

  for (i = 0; i < peaks.size() - 1; i++)
  {
    if (fabs(peaks[i].angle->angle_deg
             - peaks[i + 1].angle->angle_deg) < min_peak_dist)
    {
#ifdef DEBUG_SECONDCHORD
      if (debug) printf("peak(%i) too near at peak(%i)!\n",
                        (int) peaks[i].angle->angle_deg,
                        (int) peaks[i + 1].angle->angle_deg);
#endif
      continue;
    }

    // Check, if there is a "deep valley" between the peaks

    limit = min(peaks[i].sum_length,
                peaks[i + 1].sum_length) - min_valley_depth;

    start = min(peaks[i].position, peaks[i + 1].position);
    end = max(peaks[i].position, peaks[i + 1].position);

    for (j = start; j < end; j++)
    {
      if (hist[j].sum_length < limit)
      {
#ifdef DEBUG_SECONDCHORD
        if (debug) printf("Deep valley at %i!\n",
                          (int) hist[j].angle->angle_deg);
#endif
        *histindex = peaks[i].position;
        return true;
      }
    }
  }

  return false;
}



/****************************************************************
 * has_secondchord
 *
 * variant of has_secondchord that looks for a second chord sufficiently
 * deviating from the staff line chord
 *
 * Christoph Dalitz
 ***************************************************************/

bool has_secondchord(const vector<HistVal> &hist,
                     int min_valley_depth,
                     double center_zone, int center_level,
                     double side_zone, int side_level,
                     double min_peak_dist, int* histindex
#ifdef DEBUG_SECONDCHORD
                     , int debug = 0
#endif
                     )
{
  vector<HistVal> peaks;
  vector<HistVal>::iterator p;
  vector<HistVal>::const_iterator h;
  HistVal centerpeak;
  int lastlen;
  bool inc;
  unsigned int i, j, start, end;
  int limit;

  if (hist.empty()) throw std::runtime_error("Empty histogram!");
  *histindex = -1;

  // Run through every value of the histogram. Check
  // the chords in the side zone for a height of more
  // than *side_level*.

  for (h = hist.begin(); h != hist.end(); h++) {
    if (fabs(h->angle->angle_deg) > side_zone
        && h->sum_length * fabs(h->angle->sin) >= side_level) {
#ifdef DEBUG_SECONDCHORD
      if (debug) printf("Has long side chord at %i!\n",
                        (int) hist[i].angle->angle_deg);
#endif
      return true;
    }
  }

  // Run through every value in the histogram
  // and create a list of the maxima
  lastlen = 0;
  inc = true;
  for (h = hist.begin(), i=0; h != hist.end(); h++, i++) {
    if (inc && i>0) {
      if (h->sum_length < lastlen) {
        peaks.push_back(hist[i-1]);
        inc = false;
      }
    } else {
      if (h->sum_length > lastlen) {
        inc = true;
      }
    }
    lastlen = h->sum_length;
  }

  // If the end of the histogram remains ascending,
  // mark it as maximum, unless the first histogram
  // value is a maximum too (cyclic boundary).
  if (inc && (peaks.empty() || peaks.front().position != 0)) {
    peaks.push_back(hist.back());
  }

  // Determine the center peak stemming from the crossing staff line
  bool isfirst = true;
  for (p = peaks.begin(); p != peaks.end(); p++)
  {
    if (fabs(p->angle->angle_deg) < center_zone) {
      if (isfirst) {centerpeak = *p; isfirst = false;}
      else if (p->sum_length > centerpeak.sum_length) centerpeak = *p;
    }
  }

  if (isfirst || centerpeak.sum_length < center_level)
  {
#ifdef DEBUG_SECONDCHORD
    if (debug) printf("Has no center peak!\n");
#endif
    return false;
  }

#ifdef DEBUG_SECONDCHORD
  if (debug) {
    printf("Peak list:\n");
    for (i = 0; i < peaks.size(); i++) {
      printf("%i) %i at %lf\n",
             i, peaks[i].sum_length, peaks[i].angle->angle_deg);
    }
  }
#endif

  // look for all peaks, whether there is a valley between the peak 
  for (p = peaks.begin(); p != peaks.end(); p++) {
    if (fabs(p->angle->angle_deg
             - centerpeak.angle->angle_deg) < min_peak_dist) {
#ifdef DEBUG_SECONDCHORD
      if (debug) printf("peak(%i) too close to staff line peak(%i)!\n",
                        (int) peaks[i].angle->angle_deg,
                        (int) centerpeak.angle->angle_deg);
#endif
      continue;
    }

    // Check, if there is a "deep valley" between the peak
    // and the central staff line peak
    limit = p->sum_length - min_valley_depth;
    if (p->position > centerpeak.position) {
      start = centerpeak.position;
      end = p->position;
    } else {
      start = p->position;
      end = centerpeak.position;
    }
    for (j = start + 1; j < end; j++) {
      if (hist[j].sum_length < limit) {
#ifdef DEBUG_SECONDCHORD
        if (debug) printf("Deep valley at %i!\n",
                          (int) hist[j].angle->angle_deg);
#endif
        *histindex = p->position;
        return true;
      }
    }
  }

  return false;
}

/****************************************************************
 * rescue_secondchord
 *
 * Moves pixels from the originial image over to the rescue
 * image. For more details see the caller
 * (rescue_staffline_using_secondchord)
 *
 * Bastian Czerwinski, 2006-08-17
 ***************************************************************/

template<class T, class U>
void rescue_secondchord(T &rescue_image,
                        U &original_image,
                        int direction,
                        int threshold,
                        int max_gap_height,
                        typename T::value_type black_value,
                        int x, int y,
                        vector<HistVal> &hist,
                        int histindex)
{
        if ((direction == 1) && (histindex > 0) && 
            (fabs(hist[histindex].angle->angle_rad) > 0.35)) {
          //printf("Rescue in vector direction\n");
          // Rescue blackrun in direction of secondchord
          double height = staffline_height * 2;
          double angle = hist[histindex].angle->angle_rad;
          double dx, dy, xoff, yoff;
          Point curp;
          // walk downwards
          if (fabs(angle) < M_PI_4) {
            // rather horizontal
            if (angle > 0) dx = -1; else dx = +1;
            dy = tan(fabs(angle));
          } else {
            // rather vertical
            dy = +1;
            dx = tan(M_PI_2 + angle); // == -cot(angle)
          }
          xoff = yoff = 0.0;
          curp = Point(x, y);
          //printf("dx=%.4f dy=%.4f\n", dx, dy);
          while ((yoff < height) && (curp.y() < original_image.nrows())) {
            curp = Point((int)(x + xoff + 0.5),(int)(y + yoff + 0.5));
            if (is_white(original_image.get(curp))) break;
            rescue_image.set(curp, black_value);
            xoff += dx; yoff += dy;
          }
          // walk upwards
          if (fabs(angle) < M_PI_4) {
            // rather horizontal
            if (angle > 0) dx = +1; else dx = -1;
            dy = tan(fabs(angle));
          } else {
            // rather vertical
            dy = +1;
            dx = -tan(M_PI_2 + angle); // == cot(angle)
          }
          xoff = yoff = 0.0;
          curp = Point(x, y);
          //printf("dx=%.4f dy=%.4f\n", dx, dy);
          while ((yoff < height) && (curp.y() > 0)) {
            curp = Point((int)(x + xoff + 0.5),(int)(y - yoff - 0.5));
            if (is_white(original_image.get(curp))) break;
            rescue_image.set(curp, black_value);
            xoff += dx; yoff += dy;
          }
        }
        else {
          list<int> slice;
          list<int>::const_iterator slice_i;

          // Rescue the vertical staff line slice
          slice = get_staffline_slice(original_image, y, x,
                                      threshold, max_gap_height);
          if (slice.back() - slice.front() > threshold) {
            while (slice.size() && slice.front() < y - staffline_height / 2)
              slice.pop_front();
            while (slice.size() && slice.back() > y + staffline_height / 2)
              slice.pop_back();
          }
          for (slice_i = slice.begin(); slice_i != slice.end(); slice_i++) {
            rescue_image.set(Point(x, *slice_i), black_value);
          }         
        }
}

/****************************************************************
 * rescue_secondchord_info
 *
 * Contains parameters allowing deferred calling of the
 * rescue_secondchord method. Used in rescue_staffline_using_secondchord
 *
 * Bastian Czerwinski, 2006-08-17
 ***************************************************************/

struct rescue_secondchord_info {
  vector<HistVal> hist;
  int histindex;
  int x,y;
  bool has_secondchord;
  bool rescued;
};

/****************************************************************
 * rescue_staffline_using_secondchord
 *
 * Finds black chords running through every staff line
 * skeleton. If one of it matches certain criteria, the
 * relevant part of the staff line (vertical or in vectorfield direction,
 * depending on direction == 0 or 1) is copied to the rescue image.
 *
 * Florian Pose, 2005-09-13
 ***************************************************************/

template<class T, class U>
void rescue_stafflines_using_secondchord(T &rescue_image,
                                         U &original_image,
                                         PyObject *skel_list_py,
                                         int staffline_height,
                                         int staffspace_height,
                                         int threshold,
                                         int max_gap_height,
                                         int peak_depth,
                                         int direction,
#ifdef DEBUG_SECONDCHORD
                                         int debug,
                                         int debug_y,
                                         int debug_x,
#endif
                                         ProgressBar progress_bar
                                         = ProgressBar())
{
  typename T::value_type black_value;
  int x, histindex;
  unsigned int limit;
  list<Skeleton> skel_list;
  list<Skeleton>::const_iterator skel_i;
  list<int>::const_iterator y_i;
  ExtAngle a;
  vector<ExtAngle> thetas;
  vector<ExtAngle>::const_iterator theta_i;
  HistVal new_val;
  vector<HistVal> hist;
  deque<rescue_secondchord_info> rescue_info;

  black_value = black(rescue_image);
  limit = staffspace_height * 6;

  // Create a list containing pre-calculated angles
  for (a.angle_deg = -90.0; a.angle_deg <= 90.0; a.angle_deg += 3.0)
  {
    a.angle_rad = a.angle_deg * M_PI / 180.0;
    a.tan = tan(a.angle_rad);
    a.sin = sin(a.angle_rad);
    a.opp_angle_rad = fmod(a.angle_rad + M_PI, 2 * M_PI);
    a.opp_tan = tan(a.opp_angle_rad);
    thetas.push_back(a);
  }

  // Convert skeleton to integer list
  convert_skeleton_list(skel_list_py, skel_list);

  progress_bar.set_length(skel_list.size());

  // For every skeleton
  for (skel_i = skel_list.begin(); skel_i != skel_list.end(); skel_i++)
  {
    progress_bar.step();

    // For every point on the skeleton
    for (x = skel_i->left_x, y_i = skel_i->y_list.begin();
         y_i != skel_i->y_list.end();
         x++, y_i++)
    {
#ifdef DEBUG_SECONDCHORD
      // Skip improper lines at debugging
      if (debug_y && debug_x
          && (abs(debug_y - *y_i) > staffspace_height
              || abs(debug_x - x) > staffspace_height)) continue;
#endif

      hist.clear();

      // For every angle specified
      for (theta_i = thetas.begin(); theta_i != thetas.end(); theta_i++)
      {
        // Get the length of the black chord through the staff line
        new_val.length = chord_length(original_image, x, *y_i,
                                      theta_i->angle_rad,
                                      theta_i->tan,
                                      limit);
        new_val.opp_length = chord_length(original_image, x, *y_i,
                                          theta_i->opp_angle_rad,
                                          theta_i->opp_tan,
                                          limit);

        new_val.sum_length = new_val.length + new_val.opp_length;
        new_val.angle = &(*theta_i);
        new_val.position = hist.size();
        hist.push_back(new_val);
      }

      // lowpass filtering of histogram
      int cur_length, last_length;
      last_length = hist[0].sum_length;
      hist[0].sum_length += hist[0].sum_length + hist[1].sum_length;
      for(unsigned int i=1;i<hist.size()-1;++i) {
        cur_length = hist[i].sum_length;
        hist[i].sum_length += last_length + hist[i+1].sum_length;
        last_length = cur_length;
      }
      hist[hist.size()-1].sum_length += hist[hist.size()-1].sum_length + last_length;
      for(unsigned int i = 0; i < hist.size(); ++i) {
        hist[i].sum_length /= 3;
      }


#ifdef DEBUG_SECONDCHORD
      if (x == debug_x && *y_i == debug_y)
      {
        unsigned int i;

        printf("BEGIN GNUPLOT\n");

        for (i = 0; i < hist.size(); i++)
        {
          printf("%lf %i\n",
                 hist[i].angle->angle_deg,
                 hist[i].sum_length);
        }

        printf("END GNUPLOT\n");
      }
#endif

      // Analyze the histogram
      rescue_secondchord_info rsi;
      rsi.has_secondchord = has_secondchord(hist, peak_depth,
                                            30.0, staffline_height * 5,
                                            30.0, int(staffline_height * 1.75 + 0.5),
                                            5.0, &histindex
#ifdef DEBUG_SECONDCHORD
                                            , x == debug_x && *y_i == debug_y
#endif
                                            );
      rsi.hist = hist;
      rsi.histindex = histindex;
      rsi.x = x;
      rsi.y = *y_i;
      rsi.rescued = false;

      rescue_info.push_front(rsi);
      if( rescue_info.size() > 3 )
        rescue_info.pop_back();

      if( rescue_info.size() == 3 )
      {
        bool rescue = true;
        for( unsigned int i = 0; i < rescue_info.size(); ++i )
          if( !rescue_info[i].has_secondchord )
          {
            rescue = false;
            break;
          }

        if( rescue )
        {
          for( unsigned int i = 0; i < rescue_info.size(); ++i )
            if( !rescue_info[i].rescued )
            {
              rescue_secondchord(rescue_image,
                                 original_image,
                                 direction,
                                 threshold,
                                 max_gap_height,
                                 black_value,
                                 rescue_info[i].x, rescue_info[i].y,
                                 rescue_info[i].hist,
                                 rescue_info[i].histindex);
              rescue_info[i].rescued = true;
            }
        }
      }
    }
  }
}


/***************************************************************/

#endif
