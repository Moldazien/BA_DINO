import numpy as numpy


################################################
#
#funktion um bounding box zu berechnen, auf die zugeschnitten werden soll
#
################################################
def crop_fkt(bbox, area, percentile, width, height):
  width = width
  height = height

  width_edge = bbox[2]*percentile
  height_edge = bbox[3]*percentile

  edge = (width_edge + height_edge)

  tx = max(bbox[0]-edge, 0)
  ty = max(bbox[1]-edge, 0)
  bx = min(bbox[0]+bbox[2]+edge, width)
  by = min(bbox[1]+bbox[3]+edge, height)

  h_box = bx - tx
  w_box = by - ty

  scaling_factor = np.sqrt(area/(h_box*w_box))

  h_scale = int(np.round(h_box*scaling_factor))
  w_scale = int(np.round(w_box*scaling_factor))

  return (tx, ty, bx, by), (w_scale, h_scale), edge





################################################
#
#funktion um loss zu plotten. eingabe besteht nur aus liste von loss werten
#
################################################
import matplotlib.pyplot as plt

def plot_loss(loss, x_label='Iteration x50'):
    y_values = np.asarray(loss)
    x_values = np.arange(0,len(y_values),1)

    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.show()