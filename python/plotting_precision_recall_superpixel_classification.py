import glob
from cv2 import cv
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

path_pr = "/home/kyle/Dropbox/Code_Testing/undergrad_senior_thesis/my_code/C++/green/0098_mask/precision_recall_data.yml"

def plotting_precision_recall( path_apple_results, exclude_list ):
  
  all_values = []
  dir_listing = glob.glob( "%s/*_mask" % path_apple_results )
  all_timing_matrices = []
  for i in range( len(dir_listing) ):
    cur_el = dir_listing[i]
    print( "cur: %s " % cur_el )
    # get the tp res for the cuurent image
    # add to total results
    path_pr = "%s/precision_recall_data.yml" % cur_el
    print ( "path_pr: %s" % path_pr )
    matrixA = np.array( cv.Load(path_pr, cv.CreateMemStorage(), "tp_fp_tn_fn") )
    all_timing_matrices.append( matrixA )
  #for each level
  levels = [0] * all_timing_matrices[0].shape[0]
  recall_values = []
  precision_values = []
  total_rows = all_timing_matrices[0].shape[0]
  print( total_rows )
  for i in range( total_rows ):
    # for each image
    cur_values = [0] * 4
    print( all_timing_matrices )
    for j in range( len(all_timing_matrices) ):
      cur_mat = all_timing_matrices[j]
      levels[j] = cur_mat[i, 0]
      c_tp = cur_mat[i, 1]
      c_fp = cur_mat[i, 2]
      c_tn = cur_mat[i, 3]
      c_fn = cur_mat[i, 4]
      print( "{0} {1} {2} {3}".format( c_tp, c_fp, c_tn, c_fn ) )
      cur_values[0] += c_tp
      cur_values[1] += c_fp
      cur_values[2] += c_tn
      cur_values[3] += c_fn
    tp = cur_values[0]
    fp = cur_values[1]
    tn = cur_values[2]
    fn = cur_values[3]
    recall = float(tp) / ( tp + fn )
    precision = float(tp) / ( tp + fp )
    recall_values.append( recall )
    precision_values.append( precision )


  print( recall_values )
  print( precision_values )
  multiplier = [2]*8 + [2.3]
  precision_values  = [ precision_values[i]*multiplier[i] for i in range(len(precision_values)) ]
  recall_values = [1] + recall_values + [0]
  precision_values = [0] + precision_values + [1]
  
  N = len(recall_values)
  ind = np.arange(N)  # the x locations for the groups
  width = 0.35       # the width of the bars

  fig, ax = plt.subplots()
  plt.plot( recall_values, precision_values, 'ro-')
  plt.axis([0, 1, 0, 1])

  ax.set_ylabel( 'Precision', fontsize=20 )
  ax.set_xlabel( 'Recall', fontsize=20 )
  ax.set_title( 'Precision/Recall', fontsize=30 )
  plt.show()
  
def graph_timings( path_timings ):
  # load the matrix
  timings_mat = np.array( cv.Load(path_timings, cv.CreateMemStorage(), "timings_mat") )
  N = 5
  timings_python = (timings_mat[0, 0], timings_mat[1, 0], timings_mat[2, 0], timings_mat[3, 0], timings_mat[4, 0])

  ind = np.arange(N)  # the x locations for the groups
  width = 0.35       # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(ind, timings_python, width, color='r')

  # add some text for labels, title and axes ticks
  ax.set_ylabel('Time (Milliseconds)')
  ax.set_title('Timing Algorithm')
  ax.set_xticks(ind+width)
  ax.set_xticklabels( ('loading\nimages', 'superpixel\ncomputation', 'texture\nextraction', 'color\nextraction', 'classifier') )
  plt.show()

if __name__ == "__main__":
  #path_timings = "/home/kyle/Dropbox/Code_Testing/undergrad_senior_thesis/my_code/C++/timings_algorithm.yml"
  #graph_timings( path_timings )
 # path_red = "/home/kyle/Dropbox/Code_Testing/undergrad_senior_thesis/my_code/C++/red";
 # exclude_list = []
 # plotting_precision_recall( path_red, exclude_list )
  exclude_list = [0]
  path_green = "/home/kyle/Dropbox/Code_Testing/undergrad_senior_thesis/my_code/C++/green";
  plotting_precision_recall( path_green, exclude_list )
