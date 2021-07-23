
        
    
  #include "gameramodule.hpp"
  #include "knnmodule.hpp"

        #include "line_tracking.hpp"
  
    #include <string>
  #include <stdexcept>
  #include "Python.h"
  #include <list>

  using namespace Gamera;
  
        
    #ifndef _MSC_VER
  extern "C" {
    void init_line_tracking(void);
  }
#endif

                      static PyObject* call_extract_filled_horizontal_black_runs(PyObject* self, PyObject* args) {
            
      PyErr_Clear();
                                                                                Image* return_arg;
PyObject* return_pyarg;
                                          Image* self_arg;
PyObject* self_pyarg;
                      int windowwidth_arg;
                      int blackness_arg;
      
                                      if (PyArg_ParseTuple(args,  "Oii:extract_filled_horizontal_black_runs"
                        ,
             &self_pyarg                        ,
             &windowwidth_arg                        ,
             &blackness_arg                      ) <= 0)
           return 0;
               
              if (!is_ImageObject(self_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'self' must be an image");
          return 0;
        }
        self_arg = ((Image*)((RectObject*)self_pyarg)->m_x);
        image_get_fv(self_pyarg, &self_arg->features, &self_arg->features_len);
                                          
              try {
                      switch(get_image_combination(self_pyarg)) {
case ONEBITIMAGEVIEW:
return_arg = extract_filled_horizontal_black_runs(*((OneBitImageView*)self_arg), windowwidth_arg, blackness_arg);
break;
case CC:
return_arg = extract_filled_horizontal_black_runs(*((Cc*)self_arg), windowwidth_arg, blackness_arg);
break;
case ONEBITRLEIMAGEVIEW:
return_arg = extract_filled_horizontal_black_runs(*((OneBitRleImageView*)self_arg), windowwidth_arg, blackness_arg);
break;
case RLECC:
return_arg = extract_filled_horizontal_black_runs(*((RleCc*)self_arg), windowwidth_arg, blackness_arg);
break;
case MLCC:
return_arg = extract_filled_horizontal_black_runs(*((MlCc*)self_arg), windowwidth_arg, blackness_arg);
break;
case GREYSCALEIMAGEVIEW:
return_arg = extract_filled_horizontal_black_runs(*((GreyScaleImageView*)self_arg), windowwidth_arg, blackness_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'self' argument of 'extract_filled_horizontal_black_runs' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, ONEBIT, and GREYSCALE.", get_pixel_type_name(self_pyarg));
return 0;
}
                  } catch (std::exception& e) {
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return 0;
        }
      
                                                                                if (return_arg== nullptr) {
              if (PyErr_Occurred() == nullptr) {
                Py_XINCREF(Py_None);
                return Py_None;
               } else
                return nullptr;
            } else {
              return_pyarg = create_ImageObject(return_arg);              return return_pyarg;
            }
                              }
                static PyObject* call_thinning_v_to_skeleton_list(PyObject* self, PyObject* args) {
            
      PyErr_Clear();
                                                                                PyObject* return_arg;
PyObject* return_pyarg;
                                          Image* self_arg;
PyObject* self_pyarg;
                      int staffline_height_arg;
      
                                      if (PyArg_ParseTuple(args,  "Oi:thinning_v_to_skeleton_list"
                        ,
             &self_pyarg                        ,
             &staffline_height_arg                      ) <= 0)
           return 0;
               
              if (!is_ImageObject(self_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'self' must be an image");
          return 0;
        }
        self_arg = ((Image*)((RectObject*)self_pyarg)->m_x);
        image_get_fv(self_pyarg, &self_arg->features, &self_arg->features_len);
                            
              try {
                      switch(get_image_combination(self_pyarg)) {
case ONEBITIMAGEVIEW:
return_arg = thinning_v_to_skeleton_list(*((OneBitImageView*)self_arg), staffline_height_arg);
break;
case CC:
return_arg = thinning_v_to_skeleton_list(*((Cc*)self_arg), staffline_height_arg);
break;
case ONEBITRLEIMAGEVIEW:
return_arg = thinning_v_to_skeleton_list(*((OneBitRleImageView*)self_arg), staffline_height_arg);
break;
case RLECC:
return_arg = thinning_v_to_skeleton_list(*((RleCc*)self_arg), staffline_height_arg);
break;
case MLCC:
return_arg = thinning_v_to_skeleton_list(*((MlCc*)self_arg), staffline_height_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'self' argument of 'thinning_v_to_skeleton_list' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(self_pyarg));
return 0;
}
                  } catch (std::exception& e) {
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return 0;
        }
      
                                                              if (return_arg== nullptr) {
              if (PyErr_Occurred() == nullptr) {
                Py_XINCREF(Py_None);
                return Py_None;
               } else
                return nullptr;
            } else {
              return_pyarg = return_arg;              return return_pyarg;
            }
                              }
                static PyObject* call_skeleton_list_to_image(PyObject* self, PyObject* args) {
            
      PyErr_Clear();
                                                                                Image* return_arg;
PyObject* return_pyarg;
                                          Image* self_arg;
PyObject* self_pyarg;
                      PyObject* skeleton_list_arg;
PyObject* skeleton_list_pyarg;
      
                                      if (PyArg_ParseTuple(args,  "OO:skeleton_list_to_image"
                        ,
             &self_pyarg                        ,
             &skeleton_list_pyarg                      ) <= 0)
           return 0;
               
              if (!is_ImageObject(self_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'self' must be an image");
          return 0;
        }
        self_arg = ((Image*)((RectObject*)self_pyarg)->m_x);
        image_get_fv(self_pyarg, &self_arg->features, &self_arg->features_len);
                      skeleton_list_arg = skeleton_list_pyarg;      
              try {
                      switch(get_image_combination(self_pyarg)) {
case ONEBITIMAGEVIEW:
return_arg = skeleton_list_to_image(*((OneBitImageView*)self_arg), skeleton_list_arg);
break;
case CC:
return_arg = skeleton_list_to_image(*((Cc*)self_arg), skeleton_list_arg);
break;
case ONEBITRLEIMAGEVIEW:
return_arg = skeleton_list_to_image(*((OneBitRleImageView*)self_arg), skeleton_list_arg);
break;
case RLECC:
return_arg = skeleton_list_to_image(*((RleCc*)self_arg), skeleton_list_arg);
break;
case MLCC:
return_arg = skeleton_list_to_image(*((MlCc*)self_arg), skeleton_list_arg);
break;
case GREYSCALEIMAGEVIEW:
return_arg = skeleton_list_to_image(*((GreyScaleImageView*)self_arg), skeleton_list_arg);
break;
case GREY16IMAGEVIEW:
return_arg = skeleton_list_to_image(*((Grey16ImageView*)self_arg), skeleton_list_arg);
break;
case RGBIMAGEVIEW:
return_arg = skeleton_list_to_image(*((RGBImageView*)self_arg), skeleton_list_arg);
break;
case FLOATIMAGEVIEW:
return_arg = skeleton_list_to_image(*((FloatImageView*)self_arg), skeleton_list_arg);
break;
case COMPLEXIMAGEVIEW:
return_arg = skeleton_list_to_image(*((ComplexImageView*)self_arg), skeleton_list_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'self' argument of 'skeleton_list_to_image' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, ONEBIT, GREYSCALE, GREY16, RGB, FLOAT, and COMPLEX.", get_pixel_type_name(self_pyarg));
return 0;
}
                  } catch (std::exception& e) {
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return 0;
        }
      
                                                              if (return_arg== nullptr) {
              if (PyErr_Occurred() == nullptr) {
                Py_XINCREF(Py_None);
                return Py_None;
               } else
                return nullptr;
            } else {
              return_pyarg = create_ImageObject(return_arg);              return return_pyarg;
            }
                              }
                static PyObject* call_follow_staffwobble(PyObject* self, PyObject* args) {
            
      PyErr_Clear();
                                                                                PyObject* return_arg;
PyObject* return_pyarg;
                                          Image* self_arg;
PyObject* self_pyarg;
                      PyObject* staffline_object_arg;
PyObject* staffline_object_pyarg;
                      int staffline_height_arg;
                      int debug_arg;
      
                                      if (PyArg_ParseTuple(args,  "OOii:follow_staffwobble"
                        ,
             &self_pyarg                        ,
             &staffline_object_pyarg                        ,
             &staffline_height_arg                        ,
             &debug_arg                      ) <= 0)
           return 0;
               
              if (!is_ImageObject(self_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'self' must be an image");
          return 0;
        }
        self_arg = ((Image*)((RectObject*)self_pyarg)->m_x);
        image_get_fv(self_pyarg, &self_arg->features, &self_arg->features_len);
                      staffline_object_arg = staffline_object_pyarg;                                  
              try {
                      switch(get_image_combination(self_pyarg)) {
case ONEBITIMAGEVIEW:
return_arg = follow_staffwobble(*((OneBitImageView*)self_arg), staffline_object_arg, staffline_height_arg, debug_arg);
break;
case CC:
return_arg = follow_staffwobble(*((Cc*)self_arg), staffline_object_arg, staffline_height_arg, debug_arg);
break;
case ONEBITRLEIMAGEVIEW:
return_arg = follow_staffwobble(*((OneBitRleImageView*)self_arg), staffline_object_arg, staffline_height_arg, debug_arg);
break;
case RLECC:
return_arg = follow_staffwobble(*((RleCc*)self_arg), staffline_object_arg, staffline_height_arg, debug_arg);
break;
case MLCC:
return_arg = follow_staffwobble(*((MlCc*)self_arg), staffline_object_arg, staffline_height_arg, debug_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'self' argument of 'follow_staffwobble' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(self_pyarg));
return 0;
}
                  } catch (std::exception& e) {
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return 0;
        }
      
                                                                                                  if (return_arg== nullptr) {
              if (PyErr_Occurred() == nullptr) {
                Py_XINCREF(Py_None);
                return Py_None;
               } else
                return nullptr;
            } else {
              return_pyarg = return_arg;              return return_pyarg;
            }
                              }
                static PyObject* call_remove_line_around_skeletons(PyObject* self, PyObject* args) {
            
      PyErr_Clear();
                                                                          Image* self_arg;
PyObject* self_pyarg;
                      PyObject* skeleton_list_arg;
PyObject* skeleton_list_pyarg;
                      int staffline_height_arg;
                      int threshold_arg;
                      int max_gap_height_arg;
      
                                      if (PyArg_ParseTuple(args,  "OOiii:remove_line_around_skeletons"
                        ,
             &self_pyarg                        ,
             &skeleton_list_pyarg                        ,
             &staffline_height_arg                        ,
             &threshold_arg                        ,
             &max_gap_height_arg                      ) <= 0)
           return 0;
               
              if (!is_ImageObject(self_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'self' must be an image");
          return 0;
        }
        self_arg = ((Image*)((RectObject*)self_pyarg)->m_x);
        image_get_fv(self_pyarg, &self_arg->features, &self_arg->features_len);
                      skeleton_list_arg = skeleton_list_pyarg;                                                
              try {
                      switch(get_image_combination(self_pyarg)) {
case ONEBITIMAGEVIEW:
remove_line_around_skeletons(*((OneBitImageView*)self_arg), skeleton_list_arg, staffline_height_arg, threshold_arg, max_gap_height_arg);
break;
case CC:
remove_line_around_skeletons(*((Cc*)self_arg), skeleton_list_arg, staffline_height_arg, threshold_arg, max_gap_height_arg);
break;
case ONEBITRLEIMAGEVIEW:
remove_line_around_skeletons(*((OneBitRleImageView*)self_arg), skeleton_list_arg, staffline_height_arg, threshold_arg, max_gap_height_arg);
break;
case RLECC:
remove_line_around_skeletons(*((RleCc*)self_arg), skeleton_list_arg, staffline_height_arg, threshold_arg, max_gap_height_arg);
break;
case MLCC:
remove_line_around_skeletons(*((MlCc*)self_arg), skeleton_list_arg, staffline_height_arg, threshold_arg, max_gap_height_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'self' argument of 'remove_line_around_skeletons' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(self_pyarg));
return 0;
}
                  } catch (std::exception& e) {
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return 0;
        }
      
                                                                                                        Py_XINCREF(Py_None);
          return Py_None;
                    }
                static PyObject* call_rescue_stafflines_using_mask(PyObject* self, PyObject* args) {
            
      PyErr_Clear();
                                                                          Image* self_arg;
PyObject* self_pyarg;
                      Image* original_image_arg;
PyObject* original_image_pyarg;
                      PyObject* skeleton_list_arg;
PyObject* skeleton_list_pyarg;
                      int staffline_height_arg;
                      int threshold_arg;
      
                                      if (PyArg_ParseTuple(args,  "OOOii:rescue_stafflines_using_mask"
                        ,
             &self_pyarg                        ,
             &original_image_pyarg                        ,
             &skeleton_list_pyarg                        ,
             &staffline_height_arg                        ,
             &threshold_arg                      ) <= 0)
           return 0;
               
              if (!is_ImageObject(self_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'self' must be an image");
          return 0;
        }
        self_arg = ((Image*)((RectObject*)self_pyarg)->m_x);
        image_get_fv(self_pyarg, &self_arg->features, &self_arg->features_len);
                      if (!is_ImageObject(original_image_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'original_image' must be an image");
          return 0;
        }
        original_image_arg = ((Image*)((RectObject*)original_image_pyarg)->m_x);
        image_get_fv(original_image_pyarg, &original_image_arg->features, &original_image_arg->features_len);
                      skeleton_list_arg = skeleton_list_pyarg;                                  
              try {
                      switch(get_image_combination(self_pyarg)) {
case ONEBITIMAGEVIEW:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_mask(*((OneBitImageView*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case CC:
rescue_stafflines_using_mask(*((OneBitImageView*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_mask(*((OneBitImageView*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case RLECC:
rescue_stafflines_using_mask(*((OneBitImageView*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case MLCC:
rescue_stafflines_using_mask(*((OneBitImageView*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_mask' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
case CC:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_mask(*((Cc*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case CC:
rescue_stafflines_using_mask(*((Cc*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_mask(*((Cc*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case RLECC:
rescue_stafflines_using_mask(*((Cc*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case MLCC:
rescue_stafflines_using_mask(*((Cc*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_mask' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
case ONEBITRLEIMAGEVIEW:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_mask(*((OneBitRleImageView*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case CC:
rescue_stafflines_using_mask(*((OneBitRleImageView*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_mask(*((OneBitRleImageView*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case RLECC:
rescue_stafflines_using_mask(*((OneBitRleImageView*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case MLCC:
rescue_stafflines_using_mask(*((OneBitRleImageView*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_mask' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
case RLECC:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_mask(*((RleCc*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case CC:
rescue_stafflines_using_mask(*((RleCc*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_mask(*((RleCc*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case RLECC:
rescue_stafflines_using_mask(*((RleCc*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case MLCC:
rescue_stafflines_using_mask(*((RleCc*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_mask' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
case MLCC:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_mask(*((MlCc*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case CC:
rescue_stafflines_using_mask(*((MlCc*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_mask(*((MlCc*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case RLECC:
rescue_stafflines_using_mask(*((MlCc*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
case MLCC:
rescue_stafflines_using_mask(*((MlCc*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, threshold_arg);
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_mask' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
default:
PyErr_Format(PyExc_TypeError,"The 'self' argument of 'rescue_stafflines_using_mask' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(self_pyarg));
return 0;
}
                  } catch (std::exception& e) {
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return 0;
        }
      
                                                                                                        Py_XINCREF(Py_None);
          return Py_None;
                    }
                static PyObject* call_rescue_stafflines_using_secondchord(PyObject* self, PyObject* args) {
            
      PyErr_Clear();
                                                                          Image* self_arg;
PyObject* self_pyarg;
                      Image* original_image_arg;
PyObject* original_image_pyarg;
                      PyObject* skeleton_list_arg;
PyObject* skeleton_list_pyarg;
                      int staffline_height_arg;
                      int staffspace_height_arg;
                      int threshold_arg;
                      int max_gap_height_arg;
                      int peak_depth_arg;
                      int direction_arg;
      
                                      if (PyArg_ParseTuple(args,  "OOOiiiiii:rescue_stafflines_using_secondchord"
                        ,
             &self_pyarg                        ,
             &original_image_pyarg                        ,
             &skeleton_list_pyarg                        ,
             &staffline_height_arg                        ,
             &staffspace_height_arg                        ,
             &threshold_arg                        ,
             &max_gap_height_arg                        ,
             &peak_depth_arg                        ,
             &direction_arg                      ) <= 0)
           return 0;
               
              if (!is_ImageObject(self_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'self' must be an image");
          return 0;
        }
        self_arg = ((Image*)((RectObject*)self_pyarg)->m_x);
        image_get_fv(self_pyarg, &self_arg->features, &self_arg->features_len);
                      if (!is_ImageObject(original_image_pyarg)) {
          PyErr_SetString(PyExc_TypeError, "Argument 'original_image' must be an image");
          return 0;
        }
        original_image_arg = ((Image*)((RectObject*)original_image_pyarg)->m_x);
        image_get_fv(original_image_pyarg, &original_image_arg->features, &original_image_arg->features_len);
                      skeleton_list_arg = skeleton_list_pyarg;                                                                                          
              try {
                      switch(get_image_combination(self_pyarg)) {
case ONEBITIMAGEVIEW:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_secondchord(*((OneBitImageView*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case CC:
rescue_stafflines_using_secondchord(*((OneBitImageView*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_secondchord(*((OneBitImageView*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case RLECC:
rescue_stafflines_using_secondchord(*((OneBitImageView*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case MLCC:
rescue_stafflines_using_secondchord(*((OneBitImageView*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_secondchord' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
case CC:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_secondchord(*((Cc*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case CC:
rescue_stafflines_using_secondchord(*((Cc*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_secondchord(*((Cc*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case RLECC:
rescue_stafflines_using_secondchord(*((Cc*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case MLCC:
rescue_stafflines_using_secondchord(*((Cc*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_secondchord' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
case ONEBITRLEIMAGEVIEW:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_secondchord(*((OneBitRleImageView*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case CC:
rescue_stafflines_using_secondchord(*((OneBitRleImageView*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_secondchord(*((OneBitRleImageView*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case RLECC:
rescue_stafflines_using_secondchord(*((OneBitRleImageView*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case MLCC:
rescue_stafflines_using_secondchord(*((OneBitRleImageView*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_secondchord' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
case RLECC:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_secondchord(*((RleCc*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case CC:
rescue_stafflines_using_secondchord(*((RleCc*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_secondchord(*((RleCc*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case RLECC:
rescue_stafflines_using_secondchord(*((RleCc*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case MLCC:
rescue_stafflines_using_secondchord(*((RleCc*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_secondchord' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
case MLCC:
switch(get_image_combination(original_image_pyarg)) {
case ONEBITIMAGEVIEW:
rescue_stafflines_using_secondchord(*((MlCc*)self_arg), *((OneBitImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case CC:
rescue_stafflines_using_secondchord(*((MlCc*)self_arg), *((Cc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case ONEBITRLEIMAGEVIEW:
rescue_stafflines_using_secondchord(*((MlCc*)self_arg), *((OneBitRleImageView*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case RLECC:
rescue_stafflines_using_secondchord(*((MlCc*)self_arg), *((RleCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
case MLCC:
rescue_stafflines_using_secondchord(*((MlCc*)self_arg), *((MlCc*)original_image_arg), skeleton_list_arg, staffline_height_arg, staffspace_height_arg, threshold_arg, max_gap_height_arg, peak_depth_arg, direction_arg, ProgressBar((char *)"Rescuing parts of superimposed objects..."));
break;
default:
PyErr_Format(PyExc_TypeError,"The 'original_image' argument of 'rescue_stafflines_using_secondchord' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(original_image_pyarg));
return 0;
}
break;
default:
PyErr_Format(PyExc_TypeError,"The 'self' argument of 'rescue_stafflines_using_secondchord' can not have pixel type '%s'. Acceptable values are ONEBIT, ONEBIT, ONEBIT, ONEBIT, and ONEBIT.", get_pixel_type_name(self_pyarg));
return 0;
}
                  } catch (std::exception& e) {
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return 0;
        }
      
                                                                                                                                                                                Py_XINCREF(Py_None);
          return Py_None;
                    }
            
          static PyMethodDef _line_tracking_methods[] = {
                            {  "extract_filled_horizontal_black_runs",
          call_extract_filled_horizontal_black_runs, METH_VARARGS,
           "**extract_filled_horizontal_black_runs** (int *windowwidth*, int(1, 100) *blackness* = 75)\n\nExtract horizontal lines from an image.\n\nReturns black horizontal runs with a blackness greater than *blackness* within\na horizontal window of width *windowwidth*. In essence this is a combination\nof ``fill_horizontal_line_gaps`` and ``filter_narrow_runs(x, 'black')``.\n"        },
                        {  "thinning_v_to_skeleton_list",
          call_thinning_v_to_skeleton_list, METH_VARARGS,
           "**thinning_v_to_skeleton_list** (int *staffline_height*)\n\nSimplified thinning algorithm that only thins in the vertical\ndirection.\n\nReturn value:\n  *skeleton_list*:\n    Nested list of skeleton data. See `get_staff_skeleton_list`__ for details.\n\n.. __: musicstaves.html#get-staff-skeleton-list\n\nEach vertical black run in the image is thinned down to its middle value,\nprovided the black run is not too tall. Black runs taller than\n2 * *staffline_height* are thinned more than once.\n"        },
                        {  "skeleton_list_to_image",
          call_skeleton_list_to_image, METH_VARARGS,
           "**skeleton_list_to_image** (object *skeleton_list*)\n\nCreates an image using a skeleton list as returned by\nget_staff_skeleton_list_.\n\nArgument:\n  *skeleton_list*:\n    Nested list of skeleton data. See `get_staff_skeleton_list`__ for details.\n\n.. __: musicstaves.html#get-staff-skeleton-list\n\nThis function only helps to visualize a skeleton list. The image this\nfunction operates on is only needed for determining *ncols* and *nrows*\nof the output image. Typically the input image is the original image from\nwhich the skeletons have been extracted.\n"        },
                        {  "follow_staffwobble",
          call_follow_staffwobble, METH_VARARGS,
           "**follow_staffwobble** (object *staffline_object*, int *staffline_height* = 0, int *debug* = 0)\n\nReturns a staffline skeleton that follows a wobbling staffline.\n\nThis function is useful when you only approximately know the stafflines\n(eg. if you only know the average y position for each staffline). It\nfollows the staff line by seeking adjacent slithers of approximately\n*staffline_height*.\n\nArguments:\n\n  *staffline_object*:\n    A single staffline represented as StafflineAverage_, StafflinePolygon_ or\n    StafflineSkeleton_.\n\n.. _StafflineAverage: gamera.toolkits.musicstaves.stafffinder.StafflineAverage.html\n.. _StafflinePolygon: gamera.toolkits.musicstaves.stafffinder.StafflinePolygon.html\n.. _StafflineSkeleton: gamera.toolkits.musicstaves.stafffinder.StafflineSkeleton.html\n\n  *staffline_height*:\n    When zero, it is automatically computed as most frequent white vertical\n    runlength.\n\nReturn value:\n\n  The (hopefully) more accurate staffline represented as StafflineSkeleton__.\n\n.. __: gamera.toolkits.musicstaves.stafffinder.StafflineSkeleton.html\n\nReferences:\n\n- D. Bainbridge, T.C. Bell: *Dealing with superimposed objects in\n  optical music recognition*. http://www.cs.waikato.ac.nz/~davidb/publications/ipa97/\n\n.. note:: This postprocessing step is only reliable when staff lines\n   are absolutely solid and are not interrupted. Otherwise this method\n   can result in grossly inaccurate staff lines.\n"        },
                        {  "remove_line_around_skeletons",
          call_remove_line_around_skeletons, METH_VARARGS,
           "**remove_line_around_skeletons** (object *skeleton_list*, int *staffline_height*, int *threshold*, int *max_gap_height* = 0)\n\nRemoves a line of a certain thickness around\nevery skeleton in the given list.\n\nFor each skeleton pixel the vertical black runlength around this\npixel is determined. Is it greater than the *threshold*, a vertical\nrunlength of *staffline_height* is removed; is it smaller, the entire\nvertical black runlength is removed.\n\nThe black runlength my contain gaps of *max_gap_height* pixels. Even\nthe skeleton pixel is allowed to be white, if the surrounding gap does\nnot exceed the maximum gap height."        },
                        {  "rescue_stafflines_using_mask",
          call_rescue_stafflines_using_mask, METH_VARARGS,
           "**rescue_stafflines_using_mask** (``Image`` [OneBit] *original_image*, object *skeleton_list*, int *staffline_height*, int *threshold*)\n\nA T-shaped mask of 4 pixels is placed above every vertical\n'slice' of a staff line. If the lower pixel of the mask is\nblack and at least one of the upper 3 pixels is black, the\nvertical runlength of the staff line is rescued (i. e.\ncopied to the (actual) rescue image).\n\n- *original_image* is the image without the staff lines removed.\n\n- *skeleton*: All skeletons in a nested list.\n\n- *staffline_height*: (self-explaining)\n\n- *threshold*: The two masks are placed, *threshold* pixels apart,\n  above and below the skeleton."        },
                        {  "rescue_stafflines_using_secondchord",
          call_rescue_stafflines_using_secondchord, METH_VARARGS,
           "**rescue_stafflines_using_secondchord** (``Image`` [OneBit] *original_image*, object *skeleton_list*, int *staffline_height*, int *staffspace_height*, int *threshold*, int *max_gap_height*, int *peak_depth*, int *direction* = 0)\n\nEvery point of a staff line is examined for chords\nrunning through that point except the staff line, that are longer\nthan a certain threshold. For every chord found, the area it coveres\nis \"rescued\" from the staff line removal.\n\n- *original_image* is the image without the staff lines removed.\n\n- *skeleton*: All skeletons in a nested list.\n\n- *staffline_height*: (self-explaining)\n\n- *staffspace_height*: (self-explaining)\n\n- *threshold*: Threshold for chord detection: This represents\n  the minimum chordlength.\n\n- *peak_depth*: The minimum depth of a \"valley\" between two\n  maximums in the histogram so that they will be treated as peaks.\n\n- *direction*: when 0, the entire vertcial black run is rescued,\n  when 1, the black run in the direction of the scondchord is rescued.\n"        },
                        { nullptr }
  };
  
  static struct PyModuleDef module_line_trackingDef = {
        PyModuleDef_HEAD_INIT,
        "_line_tracking",
        nullptr,
        -1,
        _line_tracking_methods,
        nullptr,
        nullptr,
        nullptr,
        nullptr
  };


  PyMODINIT_FUNC PyInit__line_tracking(void) {
    return PyModule_Create(&module_line_trackingDef);
  }
  

