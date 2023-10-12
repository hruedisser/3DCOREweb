import copy
import warnings

import astropy.units as u
import numpy as np
import sunpy.map


def make_figure(map, image_mode, clim=[-20, 20], clip_model=True):
    '''
    Makes the main imager figure and returns the figure and axis handle.
    '''
    fig = plt.figure()
    axis = plt.subplot(projection=map)
    if image_mode == 'Plain':
        map.plot()
    else:
        map.plot(cmap='Greys_r',
                 norm=colors.Normalize(vmin=clim[0], vmax=clim[1]))
    map.draw_limb()
    yax = axis.coords[1]
    yax.set_ticklabel(rotation=90)
    if clip_model:
        axis.set_xlim([0, map.data.shape[0]])
        axis.set_ylim([0, map.data.shape[1]])

    cref = map.pixel_to_world(0*u.pix, 0*u.pix)
    if cref.Tx > 0:
        axis.invert_xaxis()
    if cref.Ty > 0:
        axis.invert_yaxis()

    return fig, axis




def plot_bodies(axis, bodies_list, smap):
    '''
    Plots in the images the possition of the pre-configured bodies (Earth, STA, Venus etc.)
    '''
    for body in bodies_list:
        body_coo = get_horizons_coord(bodies_dict[body][0], smap.date)
        if contains_coordinate(smap, body_coo):
            axis.plot_coord(body_coo, 'o', color=bodies_dict[body][1],
                            fillstyle='none', markersize=6, label=body)
            
            

def download_fits(date_process, imager, time_range=[-1, 1]):
    '''
    Downloads the imaging data (fits files) from VSO
    '''
    timerange = a.Time(date_process + datetime.timedelta(hours=time_range[0]),
                       date_process + datetime.timedelta(hours=time_range[1]))

    map_ = {}
    args = imager_dict[imager][0]
    result = Fido.search(timerange, *args)
    print(result)
    if result:
        downloaded_files = Fido.fetch(result)
        try:
            map_ = sunpy.map.Map(downloaded_files)
        except RuntimeError as err:
            print('Handling RuntimeError error:', err)
            map_ = []
        except OSError as err:
            print('Handling OSError error:', err)
            map_ = []
    else:
        map_ = []

    return map_


def maps_process(ninstr_map_in, imagers_list_in, image_mode):
    '''
    Process the images for the selected imagers and return the final maps.

    Note
    ----
    Here the ninstr_map_in is the session_state.map_ when used from the application.
    '''
    ninstr_map_out = {}
    imagers_list_out = []

    for imager in imagers_list_in:
        extra = imager_dict[imager][1]
        if imager in ninstr_map_in and ninstr_map_in[imager] != []:
            ninstr_map_out[imager] = filter_maps(ninstr_map_in[imager], extra)
            ninstr_map_out[imager] = prepare_maps(ninstr_map_out[imager], extra)
            ninstr_map_out[imager] = maps_sequence_processing(ninstr_map_out[imager],
                                                              seq_type=image_mode)
            if ninstr_map_out[imager] != []:
                imagers_list_out.append(imager)

    return ninstr_map_out, imagers_list_out

