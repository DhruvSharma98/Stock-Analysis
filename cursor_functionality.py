#mouse cursor functionality module

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Cursor(object):
    def __init__(self, ax, df):
        self.ax = ax
        self.lx = ax.axhline(alpha=0.7, color='c', dash_capstyle='round',
                             dashes=(5,2,0,2), linewidth=1)  # the horiz line
        self.ly = ax.axvline(x=df.index[0], alpha=0.7, color='c',
                             dash_capstyle='round', dashes=(5,2,0,2),
                             linewidth=1)  # the vert line

        # text location in axes coords
        self.txt = ax.text(0.7 , 0.9, '', transform=ax.transAxes)
        self.ax.figure.mpl_connect('motion_notify_event', self.mouse_move)


    def mouse_move(self, event):
        if self.ax != event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        x = mdates.num2date(x).date()
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        self.txt.set_text("date={} price={}".format(x,y))
        self.ax.figure.canvas.draw()




class SnaptoCursor(object):
    '''
    Like Cursor but the crosshair snaps to the nearest x, y point.
    For simplicity, this assumes that *x* is sorted.
    '''

    def __init__(self, ax, df, annotate_onplot, line_list=[]):
        self.ax = ax

        self.on_plot = annotate_onplot
        self.line_list = line_list
        self.annot = ax.annotate("", xy=(0,0), xytext=(5,7),
                                 textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="w"),
                                 arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        
        self.lx = ax.axhline(alpha=0.7, color='c', dash_capstyle='round',
                             dashes=(5,2,0,2), linewidth=1)  # the horiz line
        self.ly = ax.axvline(x=df.index[0], alpha=0.7, color='c',
                             dash_capstyle='round', dashes=(5,2,0,2),
                             linewidth=1)  # the vert line
        self.df = pd.DataFrame.copy(df, deep=True)
        self.df.index = self.df.index.map(mdates.date2num)

        # text location in axes coords
##        self.txt = ax.text(0.7 , 0.9, '', transform=ax.transAxes)
        self.lx.set_visible(False)
        self.ly.set_visible(False)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.mouse_move)

    def mouse_move(self, event):
        if event.inaxes != self.ax:
            self.lx.set_visible(False)
            self.ly.set_visible(False)
            self.annot.set_visible(False)
            self.ax.figure.canvas.draw_idle()

        else:
            x, y = event.xdata, event.ydata
            if x >= self.df.index[0] and x<=self.df.index[self.df.shape[0]-1]:
                snap_x = min(min(self.df.index[self.df.index >= x]),
                             max(self.df.index[self.df.index < x]))
                pos = [snap_x,y]
                flag=False
                for curve in self.ax.get_lines():
                    if curve.get_gid() in self.line_list:
                        if curve.contains(event)[0]:
                            flag=True
                            break
                        else:
                            continue
                self.annotate(event, pos, flag)

                # update the line positions
                self.lx.set_ydata(y)
                self.ly.set_xdata(snap_x)
                self.lx.set_visible(True)
                self.ly.set_visible(True)
##                self.txt.set_text("date={} price={}".format(x,y))
                self.ax.figure.canvas.draw_idle()

    def update_annot(self, pos):
        self.annot.xy = pos
        x,y = pos
        x = mdates.num2date(x).date()
        y = '{0:.2f}'.format(y)
        text = "{}, {}".format(x,y)
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor('k')
        self.annot.get_bbox_patch().set_alpha(0.4)


    def annotate(self, event, pos, flag):
        if self.on_plot:
            if flag:
                self.update_annot(pos)
                self.annot.set_visible(True)
                self.ax.figure.canvas.draw_idle()
            else:
                self.annot.set_visible(False)
                self.ax.figure.canvas.draw_idle()
        else:
            if event.inaxes == self.ax:
                self.update_annot(pos)
                self.annot.set_visible(True)
                self.ax.figure.canvas.draw_idle()
