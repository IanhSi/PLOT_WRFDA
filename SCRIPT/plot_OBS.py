from __future__ import print_function

import os
import sys
import datetime
import re
import time
from glob       import glob
from pathlib    import Path 

from pandarallel import pandarallel

import h5py
import cmaps
import numpy                as np
import pandas               as pd
import xarray               as xr

import metpy
from metpy.units    import units
from metpy.calc     import specific_humidity_from_dewpoint

import scipy.ndimage        as ndimage
import matplotlib
import matplotlib.pyplot    as plt
import matplotlib.colors    as col
import matplotlib.ticker    as ticker
import matplotlib.dates     as mdates
from matplotlib.backends.backend_pdf import PdfPages

import cartopy.crs              as ccrs   
import cartopy.feature          as cfeat 
import cartopy.io.shapereader   as shpreader
from   cartopy.mpl.ticker    import LongitudeFormatter,LatitudeFormatter 
from   cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import salem
import rioxarray
def save_pic(fig=None, savepath=None, savename=None,if_resave=True):
    savefile=os.path.join(savepath,savename)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(savefile):
        print('saving pic: '+savefile)
        try:
            fig.savefig(savefile, bbox_inches='tight')
        except:
            fig.savefig(savefile)
        plt.close()
    else:
        if if_resave:
            print('re-saving pic: '+savefile)
            os.remove(savefile)
            try:
                fig.savefig(savefile, bbox_inches='tight')
            except:
                fig.savefig(savefile)
            plt.close()
        else:
            print(savefile,' exist')

def draw_contour_map(fig,ax,lats,lons,var,data_proj,plot_proj,levels,cmap,norm,lat_s=26,lat_e=34.5,lon_s=97.2,lon_e=108.7,tick_inv=2):
    # process lat lon
    if lons.ndim == 2 and lats.ndim == 2:
        lon2d, lat2d = lons, lats
    else:
        lon2d, lat2d = np.meshgrid(lons, lats)

    # add shp
    PATH_SRC = r'/public/home/ipm_zhengq/local/SHPFILE'
    shpfile  = str(Path(PATH_SRC) / r'Province.shp')
    # shpfile  = str(Path(PATH_SRC) / r'SiChuan_Province.shp')
    sc_shp   = list(shpreader.Reader(shpfile).geometries())
    ax.add_geometries(sc_shp, ccrs.PlateCarree(), edgecolor='k', linewidth=1.2, linestyle='-', facecolor='None')

    shpfile  = str(Path(PATH_SRC) / r'Sichuan.shp')
    sw_shp   = list(shpreader.Reader(shpfile).geometries())
    # ax.add_geometries(sw_shp, ccrs.PlateCarree(), edgecolor='b', linewidth=1.2, facecolor='None')

    # ticks
    ax.set_xticks(np.arange(int(lon_s),lon_e,tick_inv),  crs=plot_proj)
    ax.set_yticks(np.arange(int(lat_s),lat_e,tick_inv),  crs=plot_proj)
    ax.set_extent([lon_s, lon_e, lat_s, lat_e],  crs=plot_proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=18)

    # left title
    try:
        ax.set_title(var.long_name, loc='left', fontsize='18')
    except:
        try:
            ax.set_title(var.description,loc='left', fontsize='18')
        except:
            print('no description')

    # right tile
    try:
        ax.set_title(pd.to_datetime(var.time.values).strftime("%Y-%m-%d %H:%M:%S"),loc='right',fontsize='18')
    except:
        try:
            ax.set_title(pd.to_datetime(var.Time.values).strftime("%Y-%m-%d %H:%M:%S"),loc='right',fontsize='18')
        except:
            print('no time information')

    # plot
    ac   = ax.contourf(lon2d,lat2d,var,levels=levels,norm=norm,cmap=cmap,extend='both',transform=data_proj)

    # add colorbar
    l,b,w,h = 0.25, 0.01, 0.5, 0.03
    rect = [l,b,w,h]
    cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(ac, cax = cbar_ax,orientation='horizontal',spacing='proportional')
    try:
        cb.set_label(var.units,loc='center',fontsize=18)
    except:
        print('no units')
    cb.ax.tick_params(labelsize=18)
    # cb.formatter.set_powerlimits((0, 0))
    return ax

def plot_radiance_location():
    lat_s                   = 15.0
    lat_e                   = 42.5
    lon_s                   = 75.0
    lon_e                   = 120.5
    channels                = [11,12,13,14,15]

    filepath_diag           = r'/public/home/ipm_zhengq/local/git/PLOT_WRFDA/RAD_DIAGS'
    filename_diag           = r'diags_fy3-4-mwhs2_2022071906.nc'
    filepath_ana            = r'/public/home/ipm_zhengq/local/SWC_WINGS_WRF/OUT_NARUN/SWC_WINGS_3KM/2022071906/WRFDA'
    filename_ana            = r'wrfvar_output'
    savepath                = r'/public/home/ipm_zhengq/local/git/PLOT_WRFDA/PIC'
    


    # --- open file and get data
    file                    = os.path.join(filepath_diag,filename_diag)
    ds_diag                 = xr.open_dataset(file,engine='netcdf4')
    file                    = os.path.join(filepath_ana,filename_ana)
    ds_ana                  = xr.open_dataset(file,engine='netcdf4')
    OMB                     = ds_diag['tb_inv']
    TER                     = ds_ana['HGT'].squeeze()

    # --- loop for channel
    for channel in channels:
        fig                     = plt.figure(figsize=(12,6),dpi=150)
        ax                      = fig.add_subplot(111,projection = ccrs.PlateCarree())

        # --- draw terrian
        var                     = TER.copy()
        lats                    = TER.coords['XLAT']
        lons                    = TER.coords['XLONG']

        # colorbar
        levels                  = np.arange(1,25,1)*200.0 
        cmap                    = cmaps.MPL_gist_yarg
        idx                     = np.round(np.linspace(20, cmap.N - 1-20, len(levels) + 1)).astype(int)    # extend=both
        colors                  = cmap(idx)
        colormap, norm          = col.from_levels_and_colors(levels, colors, extend='both')
        ax                      = draw_contour_map(fig,ax,lats,lons,var,ccrs.PlateCarree(),ccrs.PlateCarree(),levels,colormap,norm,
                                                    lat_s=lat_s,lat_e=lat_e,lon_s=lon_s,lon_e=lon_e,tick_inv=5)
        ax.set_title('Channel: {}'.format(str(channel)),loc='right',fontsize='20')

        # --- draw OMB
        var                     = ds_diag['tb_inv'].data[:,channel-1]
        qc                      = ds_diag['tb_qc'].data[:,channel-1]
        var                     = np.where(qc==1,var,np.nan)
        lats                    = ds_diag['lat']
        lons                    = ds_diag['lon']

        # colorbar
        levels                  = np.arange(-5,6,1)*1.0   # PW
        cmap                    = cmaps.BlueWhiteOrangeRed
        idx                     = np.round(np.linspace(0, cmap.N - 1, len(levels) + 1)).astype(int)    # extend=both
        colors                  = cmap(idx)
        colormap, norm          = col.from_levels_and_colors(levels, colors, extend='both')

        ac                      = ax.scatter(lons, lats, c=var, s=5, cmap=colormap, norm=norm)
        l,b,w,h                 = 0.85, 0.2, 0.02, 0.6
        rect                    = [l,b,w,h]
        cbar_ax                 = fig.add_axes(rect)
        cb                      = fig.colorbar(ac, cax = cbar_ax,orientation='vertical',spacing='proportional')
        cb.set_label('K',loc='center',rotation='horizontal',fontsize=18)
        cb.ax.tick_params(labelsize=18)

        # --- save pic
        savename                = r'radiance_location_channel{}.png'.format(str(channel))
        save_pic(fig,savepath,savename)


def main():
    # print(matplotlib.matplotlib_fname())
    # plt.rc('font',family='Times New Roman')
    plt.rc('font',family='Arial')
    plot_radiance_location()


def test():
    # filepath_ERA5       = r'/public/home/ipm_zhengq/DATAs/REANALYSIS/ERA5/hourly'
    # date_obs_s          = '2022-07-18'
    # time_obs_s          = '08:03:30'
    # datetime_start_str  = '{} {}'.format(date_obs_s,time_obs_s)
    # wrf_file              = r'/public/home/ipm_zhengq/SWC3KM/MODEL/SWC-TSPAS/RAWOUT/SWC_WINGS_3KM/2021061500/WRF/WRFO5/wrfout_d01_2021-06-15_03:00:00'
    # ds                    = salem.open_wrf_dataset(wrf_file)
    # print(ds)
    loc_1d=np.array(34)
    loc_2d=np.random.normal(loc=30, scale=10, size=100).reshape((10,10))



if __name__ == '__main__':
    main()
    # test()
