title = 'baygaudviewer Mk.Ia'

# 22.09.16.
# Minsu Kim @ Sejong Univ
# mandu447@gmail.com

# DEPENDENCIES
# sudo apt install python-tk
# pip3 install numpy
# pip3 install matplotlib
# pip3 install astropy
# pip3 install spectral_cube

import glob
import os
import sys
from tkinter import *
from tkinter import filedialog, messagebox, ttk

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from spectral_cube import SpectralCube

dict_params = {'cursor_xy':(-1,-1), 'multiplier_cube':1000.0, 'unit_cube':r'mJy$\,$beam$^{-1}$', 'multiplier_spectral_axis':0.001}
dict_data = {}
dict_obj  = {}
dict_plot = {'fix_cursor':False}
plt.rcParams["hatch.linewidth"] = 4
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']

def gaussian(x, amp, vel, disp):
    val = amp * np.exp(-np.power(x-vel, 2.) / (2 * np.power(disp, 2.)))
    return val

def colorbar(img, spacing=0, cbarwidth=0.01, orientation='vertical', pos='right', label='', ticks=[0], fontsize=13):

    ax = img.axes
    fig = ax.figure
    if(orientation=='vertical'):
        if(pos=='right'):
            cax = fig.add_axes([ax.get_position().x1+spacing, ax.get_position().y0, cbarwidth, ax.get_position().height])
        elif(pos=='left'):
            cax = fig.add_axes([ax.get_position().x0-spacing-cbarwidth, ax.get_position().y0, cbarwidth, ax.get_position().height])
            cax.yaxis.set_ticks_position('left')
    elif(orientation=='horizontal'):
        if(pos=='top'):
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y1+spacing, ax.get_position().width, cbarwidth])
            cax.tick_params(axis='x', labelbottom=False, labeltop=True)

            # cax.xaxis.tick_top()
        elif(pos=='bottom'):
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-spacing-cbarwidth, ax.get_position().width, cbarwidth])
    
    if(len(ticks)!=1):
        cbar = plt.colorbar(img, cax=cax, orientation=orientation, ticks=ticks)
    else: cbar = plt.colorbar(img, cax=cax, orientation=orientation)
    cbar.set_label(label=label, fontsize=fontsize)
    return cbar, cax

def label_panel(ax, text, xpos=0.05, ypos=0.95, color='black', fontsize=10, inside_box=False, pad=5.0):
    # MAKES A LABEL ON GIVEN PANEL

    # PARAMETERS:
    # ax: matplotlib ax
    # text: (str) message to write
    # xpos: xpos of labelbox (relative corrdinates, 0 to 1)
    # ypos: ypos of labelbox (relative corrdinates, 0 to 1)
    # color: color of the text
    # fontsize=: fontsize of the text
    # inside_box = whether to write the text inside a box
    # pad: space between the text and the surrounding box

    # RETURNS:
    # Nothing

    if(inside_box==True):
        ax.text(xpos, ypos, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top', 
            bbox=dict(facecolor='none', edgecolor=color, pad=pad))
    else:
        ax.text(xpos, ypos, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top', 
            bbox=dict(facecolor='none', edgecolor='none', pad=pad))

def fillentry(entry, content):
    if(entry['state']=='readonly'):
        entry['state']='normal'
        entry.delete(0, "end")
        entry.insert(0, content)
        entry['state']='readonly'
    else:
        entry.delete(0, "end")
        entry.insert(0, content)

def makelabelentry(frame, array, title, startcol, widthlabel, widthentry):
    if(len(title)==0):
        title=array
    for i, content in enumerate(array):
        globals()['label_%s'%(content)] = Label(frame, text=title[i], width=widthlabel, anchor='e')
        globals()['label_%s'%(content)].grid(row=i+startcol, column=0, padx=5)
        globals()['entry_%s'%(content)] = Entry(frame, width=widthentry, justify='right')
        globals()['entry_%s'%(content)].grid(row=i+startcol, column=1)


def initdisplay():

    def _clear(canvas):
        for item in canvas.get_tk_widget().find_all():
            canvas.get_tk_widget().delete(item)

    if 'fig1' not in dict_plot:

        fig1, ax1 = plt.subplots()#tight_layout=True)
        fig1.set_figwidth(500/fig1.dpi)
        fig1.set_figheight(460/fig1.dpi)
        fig1.subplots_adjust(left=0.1, right=0.90, top=0.99, bottom=0.05)

        canvas1 = FigureCanvasTkAgg(fig1, master=dict_obj['frame_display'])   #DRAWING FIGURES ON GUI FRAME
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=TOP)#, fill=BOTH, expand=True)
        fig1.canvas.mpl_connect('motion_notify_event', tracecursor)  #CONNECTING MOUSE CLICK ACTION
        fig1.canvas.mpl_connect('scroll_event', zoom)

        fig2, (ax2, ax3) = plt.subplots(nrows=2, sharex=True)
        fig2.set_figwidth(500/fig2.dpi)
        fig2.set_figheight(500/fig2.dpi)
        fig2.subplots_adjust(hspace=0, top=0.96, bottom=0.16)

        ax2.plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))

        canvas2 = FigureCanvasTkAgg(fig2, master=dict_obj['frame_line'])
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        dict_plot['fig1']    = fig1
        dict_plot['ax1']     = ax1
        dict_plot['canvas1'] = canvas1

        dict_plot['fig2']    = fig2
        dict_plot['ax2']     = ax2
        dict_plot['ax3']     = ax3
        dict_plot['canvas2'] = canvas2

        dict_params['drawnew'] = False

    elif(dict_params['drawnew']):
        del dict_plot['img1']
        del dict_plot['fig1']

        dict_obj['frame_display'].destroy()
        dict_obj['frame_display'] = Frame(frame_L, height=500, width=500, bg='white')
        dict_obj['frame_display'].pack(side='top')

        dict_obj['frame_mapselect'].pack_forget()
        dict_obj['frame_mapselect'].pack(fill=BOTH, expand=True)

        dict_obj['frame_line'].destroy()
        dict_obj['frame_line'] = Frame(frame_R, width=500,height=500, bg='white')
        dict_obj['frame_line'].pack()

        initdisplay()

    if 'img1' in dict_plot:
        cur_xlim = dict_plot['ax1'].get_xlim()
        cur_ylim = dict_plot['ax1'].get_ylim()
        dict_plot['ax1'].clear()
        dict_plot['ax1'].set_xlim(cur_xlim)
        dict_plot['ax1'].set_ylim(cur_ylim)
    else:
        dict_plot['ax1'].clear()
    
    if 'cax' in dict_plot:
        dict_plot['cax'].clear()
        del dict_plot['cax']
    path_map = glob.glob(dict_params['path_fig1'])[0]
    dict_plot['img1'] = dict_plot['ax1'].imshow(fits.getdata(path_map), interpolation='none')
    ylim = dict_plot['ax1'].get_ylim()
    if(ylim[0]>ylim[1]):
        dict_plot['ax1'].invert_yaxis()
    if(dict_plot['fix_cursor']):
        cross = dict_plot['ax1'].scatter(dict_params['cursor_xy'][0], dict_params['cursor_xy'][1], marker='+', color='white', s=100)
        cross.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='black')])
    _,dict_plot['cax'] = colorbar(dict_plot['img1'], cbarwidth=0.03)
    dict_plot['canvas1'].draw()

    dict_plot['ax2'].clear()
    dict_plot['ax3'].clear()
    
    # dict_plot['ax2'].plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))
    # dict_plot['ax3'].plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))

    dict_plot['canvas2'].draw()

    drawplots()


    # plt.close(fig)


def readdata(path_cube=None, path_classified=None):

    if(path_cube!=None):
        dict_params['path_cube'] = path_cube
    if(path_classified!=None):
        dict_params['path_classified'] = path_classified

    dict_params['path_fig1'] = dict_params['path_classified']+'/G*g01/*5.fits'

    dict_data['cube'] = fits.getdata(dict_params['path_cube'])*dict_params['multiplier_cube']
    dict_data['spectral_axis'] = SpectralCube.read(dict_params['path_cube']).spectral_axis.value*dict_params['multiplier_spectral_axis']

    dict_data['imsize'] = dict_data['cube'][0,:,:].shape

    n_gauss = len(glob.glob(dict_params['path_classified']+"/G0*/"))

    amps   = np.empty(n_gauss, dtype=object)
    vels   = np.empty(n_gauss, dtype=object) #3
    disps  = np.empty(n_gauss, dtype=object) #2

    bg = fits.getdata(glob.glob(dict_params['path_classified']+'/G0*g01/*.0.fits')[0])
    data_noise = fits.getdata(glob.glob(dict_params['path_classified']+'/G0*g01/*6.fits')[0])

    dict_data['noise'] = data_noise

    for i in range(n_gauss):
        # name_bg    = glob.glob(path_classified+'/G0*g0{}/*.0.fits'.format(i+1))[0]
        name_psn   = glob.glob(dict_params['path_classified']+'/G0*g0{}/*7.fits'.format(i+1))[0]
        name_vel   = glob.glob(dict_params['path_classified']+'/G0*g0{}/*3.fits'.format(i+1))[0]
        name_disp  = glob.glob(dict_params['path_classified']+'/G0*g0{}/*2.fits'.format(i+1))[0]

        vels[i]   = fits.getdata(name_vel)
        disps[i]  = fits.getdata(name_disp)

        data_psn   = fits.getdata(name_psn)
        # amps[i] = bgs[i] + (data_psn * data_noise)
        # if(bgsub==True): amps[i] = data_psn * data_noise
        # else: amps[i] = data_psn * data_noise + bg
        amps[i] = data_psn * data_noise# + bg

        del data_psn
    del data_noise

    dict_data['amps']  = amps
    dict_data['vels']  = vels
    dict_data['disps'] = disps
    dict_data['bg']    = bg*dict_params['multiplier_cube']

    dict_params['drawnew'] = True
    initdisplay()


def loaddata():

    def browse_cube():
        path_cube = filedialog.askopenfilename(title='Path to cube', filetypes=[('FITS file', '.fits .FITS')])
        if(len(path_cube)==0): return

        if(len(fits.getdata(path_cube).shape)<3 or len(SpectralCube.read(path_cube).spectral_axis)==1):
            messagebox.showerror("Error", "Cube should have at least three dimensions.")
            return
        
        fillentry(entry_path_cube, path_cube)

        possible_path_classified = glob.glob(os.path.dirname(path_cube)+'/baygaud_output*/output_merged/classified*')
        if(len(possible_path_classified)==1):
            browse_classified(possible_path_classified[0])
        elif(len(possible_path_classified)>1):
            browse_classified(initialdir=os.path.dirname(possible_path_classified[0]))

    def browse_classified(path_classified=None, initialdir=None):
        if(path_classified==None):
            path_classified = filedialog.askdirectory(title='Path to classified directory', initialdir=initialdir)
            if(len(path_classified)==0): return

        ifexists = os.path.exists(path_classified+"/single_gfit")
        if(ifexists==False):
            messagebox.showerror("Error", "No proper data found inside.")
            return

        fillentry(entry_path_classified, path_classified)  

    def btncmd_toplv_browse_cube():
        browse_cube()

    def btncmd_toplv_browse_classified():
        browse_classified()



    def btncmd_toplv_apply():
        dict_params['path_cube'] = entry_path_cube.get()
        dict_params['path_classified'] = entry_path_classified.get()
        readdata()

        dict_plot['toplv'].destroy()
   

    def btncmd_toplv_cancel():
        toplv.destroy()

    toplv = Toplevel(root)

    frame_toplv1 = Frame(toplv)
    frame_toplv2 = Frame(toplv)

    makelabelentry(frame_toplv1, ['path_cube', 'path_classified'], [], 0, 20, 20)

    btn_toplv_browsecube = Button(frame_toplv1, text='Browse', command=btncmd_toplv_browse_cube)
    btn_toplv_browsecube.grid(row=0, column=2)

    btn_toplv_browseclassified = Button(frame_toplv1, text='Browse', command=btncmd_toplv_browse_classified)
    btn_toplv_browseclassified.grid(row=1, column=2)

    ttk.Separator(frame_toplv2, orient='horizontal').pack(fill=BOTH)

    btn_toplv_apply = Button(frame_toplv2, text='Apply', command=btncmd_toplv_apply)
    btn_toplv_cancel = Button(frame_toplv2, text='Cancel', command=btncmd_toplv_cancel)
    btn_toplv_cancel.pack(side='right')
    btn_toplv_apply.pack(side='right')

    frame_toplv1.pack()
    frame_toplv2.pack(fill=BOTH)

    dict_plot['toplv'] = toplv


def apply_mapselect(*args):

    var = dict_plot['var_mapselect'].get()
    # ['Integrated flux', 'SGfit velocity', 'SGfit vdisp', 'Ngauss', 'SGfit Peak S/N']

    if(var=='Integrated flux'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/G*g01/*all*fits'
    if(var=='SGfit velocity'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/single_gfit/*3.fits'
    if(var=='SGfit vdisp'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/single_gfit/*2.fits'
    if(var=='Ngauss'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/G*g01/*5.fits'
    if(var=='SGfit Peak S/N'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/single_gfit/*7.fits'

    initdisplay()

def fix_cursor(event):
    dict_plot['fix_cursor'] = (dict_plot['fix_cursor']+1)%2

    initdisplay()

root = Tk()

root.title(title)
# root.bind("<Return>", lambda x: updatedisplay())
root.resizable(False, False)


menubar = Menu(root)

menu_1 = Menu(menubar, tearoff=0)
menu_1.add_command(label="Load data", command=loaddata)

menu_2 = Menu(menubar, tearoff=0)
menu_2.add_command(label='TBU')

menubar.add_cascade(label="Load", menu=menu_1)
menubar.add_cascade(label="Option", menu=menu_2)

dict_obj['frame_master'] = Frame(root)
frame_L = Frame(dict_obj['frame_master'], height=500, width=500, bg='white')
frame_M = Frame(dict_obj['frame_master'], height=500, width=50, bg='white')
frame_R = Frame(dict_obj['frame_master'], height=500, width=500, bg='white')

dict_obj['frame_display'] = Frame(frame_L, height=500, width=500, bg='white')
dict_obj['frame_display'].pack()

dict_obj['frame_mapselect'] = Frame(frame_L)
OptionList = ['Integrated flux', 'SGfit velocity', 'SGfit vdisp', 'Ngauss', 'SGfit Peak S/N']
dict_plot['var_mapselect'] = StringVar()
dict_plot['var_mapselect'].set(OptionList[3])

dropdown_mapselect = OptionMenu(dict_obj['frame_mapselect'], dict_plot['var_mapselect'], *OptionList)
# dropdown_mapselect.config
dropdown_mapselect.pack(side='right')
dict_plot['var_mapselect'].trace("w", apply_mapselect)
dict_obj['frame_mapselect'].pack(fill=BOTH, expand=True)

dict_obj['frame_line'] = Frame(frame_R, width=500,height=500, bg='white')
dict_obj['frame_line'].pack()

frame_L.pack(fill=BOTH, expand=True, side='left')
frame_M.pack(fill=BOTH, expand=True, side='left')
frame_R.pack(fill=BOTH, expand=True, side='right')
dict_obj['frame_master'].pack(fill=BOTH, expand=True)

def drawplots():

    # fig, ax = plt.subplots()#tight_layout=True)
    # fig.set_figwidth(500/fig.dpi)
    # fig.subplots_adjust(left=0.1, right=0.90, top=0.99, bottom=0.05)

    # path_map = glob.glob(dict_params['path_classified']+'/single_gfit/*5.fits')[0]
    # img = ax.imshow(fits.getdata(path_map))
    # ax.invert_yaxis()
    # colorbar(img)

    # canvas = FigureCanvasTkAgg(fig, master=frame_display)   #DRAWING FIGURES ON GUI FRAME
    # canvas.draw()
    # canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
    # fig.canvas.mpl_connect('motion_notify_event', tracecursor)  #CONNECTING MOUSE CLICK ACTION

    x,y=dict_params['cursor_xy']
    dict_plot['ax2'].clear()
    dict_plot['ax3'].clear()
    # dict_plot['ax2'].plot(dict_data['spectral_axis'], dict_data['cube'][:,y,x])

    dict_plot['ax2'].fill_between(dict_data['spectral_axis'], dict_data['cube'][:,y,x], hatch=r'//', color='lightgray', edgecolor='white')
    dict_plot['ax2'].plot(        dict_data['spectral_axis'], dict_data['cube'][:,y,x], color='lightgray')
    
    subed = np.full_like(dict_data['cube'][:,y,x], dict_data['cube'][:,y,x])
    total = np.zeros_like(dict_data['spectral_axis'])

    for i in range(len(dict_data['vels'])):
        vel  = dict_data['vels'][i][y,x]
        disp = dict_data['disps'][i][y,x]
        amp  = dict_data['amps'][i][y,x]
        bg   = dict_data['bg'][y,x]

        if(np.any(np.isnan([bg,vel,disp,amp]))): continue

        ploty = gaussian(dict_data['spectral_axis'], amp,vel,disp)*dict_params['multiplier_cube'] + bg
        # ploty+=dict_data['bg'][y,x]*bgsub
        total += (ploty-bg)

        dict_plot['ax2'].plot(dict_data['spectral_axis'], ploty, label='BAYGAUD G{}'.format(i+1), color=colors[i], alpha=0.5)

    
    total += bg
    subed -= total

    dict_plot['ax2'].plot(dict_data['spectral_axis'], total, color='black', ls=':', alpha=0.7)

    dict_plot['ax3'].fill_between(dict_data['spectral_axis'], subed, hatch=r'//', color='lightgray', edgecolor='white')
    dict_plot['ax3'].plot(        dict_data['spectral_axis'], subed, color='lightgray')

    label_panel(dict_plot['ax2'], '(x,y)=({},{})'.format(x,y))
    label_panel(dict_plot['ax3'], 'Residuals')

    # dict_plot['ax2'].set_ylabel('Flux density ({})'.format(dict_params['unit_cube']))
    dict_plot['ax2'].text(-0.12, -0, 'Flux density ({})'.format(dict_params['unit_cube']), ha='center', va='center', transform = dict_plot['ax2'].transAxes, rotation=90)
    dict_plot['ax3'].set_xlabel(r'Spectral axis (km$\,$s$^{-1}$)')

    # canvas_line1 = FigureCanvasTkAgg(fig2, master=frame_line1)
    dict_plot['canvas2'].draw()




def tracecursor(event):
    if(dict_plot['fix_cursor']==False):
        # x,y=event.x, event.y
        if event.inaxes:
            # ax = event.inaxes
            cursor_xy = (round(event.xdata),round(event.ydata))
            # print(cursor_x, cursor_y)

            if(dict_params['cursor_xy']==cursor_xy[0] and dict_params['cursor_xy'][1]==cursor_xy[1]):
                return
            else:
                dict_params['cursor_xy']=cursor_xy
                drawplots()   

def zoom(event):
    cur_xlim = dict_plot['ax1'].get_xlim()
    cur_ylim = dict_plot['ax1'].get_ylim()

    xdata = event.xdata # get event x location
    ydata = event.ydata # get event y location

    base_scale = 2.

    if event.button == 'up':
        # deal with zoom in
        scale_factor = 1 / base_scale
    elif event.button == 'down':
        # deal with zoom out
        scale_factor = base_scale
    else:
        # deal with something that should never happen
        scale_factor = 1
        print(event.button)

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

    dict_plot['ax1'].set_xlim(np.max([xdata - new_width * (1-relx),0]), np.min([xdata + new_width * (relx),dict_data['imsize'][1]-1]))
    dict_plot['ax1'].set_ylim(np.max([ydata - new_height * (1-rely),0]), np.min([ydata + new_height * (rely),dict_data['imsize'][0]-1]))
    # ax.figure.canvas.draw()
    dict_plot['canvas1'].draw()



root.config(menu=menubar)
root.bind('f', fix_cursor)
# print(sys.argv)
if(len(sys.argv)>1):
    readdata(path_cube=sys.argv[1], path_classified=sys.argv[2])

root.mainloop()

