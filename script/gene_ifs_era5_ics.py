"""
Description: script to download and gets ensemble members from ecmwf ifs_en forecast model

"""

import os, pygrib, zarr, argparse
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import cdsapi
from collections import defaultdict
import yaml

class ERA5DataProcessor:
    def __init__(self,pdate,member,download_directory):
        self.pdate = pdate
        self.member = member
        self.cache_dir  = download_directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # define the variables
        self.param_sfc = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature", 
                          "2m_temperature","mean_sea_level_pressure","surface_pressure","geopotential", 
                          "land_sea_mask", "skin_temperature"]
        self.param_sfc1 = ["10u","10v","2d","2t","msl","sp","z","lsm","skt"]             
        self.para_pl = ["geopotential","specific_humidity","temperature","u_component_of_wind",
                        "v_component_of_wind","vertical_velocity"] 
        self.para_pl1 = ["z","q","t","u","v","w"]                
        self.pl_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        
    def era5_surf_download(self,t0): 
        new_str = t0.strftime("%Y%m%d_%H")
        fout = f'{self.cache_dir}/era5_surface_variable_{new_str}.grib'
        if not os.path.exists(fout):
            c = cdsapi.Client()
            r = c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        "product_type": 'ensemble_members',
                        "variable": self.param_sfc,
                        "year": f"{t0.year:04d}",
                        "month": f"{t0.month:02d}",
                        "day": f"{t0.day:02d}",
                        "time": f"{t0.hour:02d}",
                        'format': 'grib'
                    },
                )
            r.download(fout)
        return fout

    def era5_pl_download(self,t0):
        new_str = t0.strftime("%Y%m%d_%H")
        fout = f'{self.cache_dir}/era5_pressure_level_variable_{new_str}.grib'
        if not os.path.exists(fout):
            c = cdsapi.Client()
            r = c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        "product_type": 'ensemble_members',
                        "variable": self.para_pl,
                        "year": f"{t0.year:04d}",
                        "month": f"{t0.month:02d}",
                        "day": f"{t0.day:02d}",
                        "time": f"{t0.hour:02d}",
                        "pressure_level": self.pl_levels,
                        'format': 'grib'
                    },
                )
            r.download(fout)
        return fout

    def p050tp100(self,data,plat,plon,method):
        new_lat = np.arange(90, -90.001, -1); new_lon = np.arange(0, 360, 1);
        da = xr.DataArray(data, dims=("lat", "lon"),coords={"lat":plat,"lon":plon})
        da_new = da.interp(lat=new_lat, lon=new_lon, method=method, kwargs={"fill_value": "extrapolate"})
        da_new = np.array(da_new)
        # the ufs2arco generated training: [lon,lat] with lon [0,360] and lat [-90,90]
        # while, era5 is [lat,lon] with lon [0,360] but lat [90,-90]
        da_new  = np.flip(da_new, axis=0).T  # convert lat to [-90,90] and dim[lon,lat]
        return da_new

    def fill_nan_linear(self,arr):
        nans = np.isnan(arr)
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        return arr

    def get_data(self,params,levelist=[]):
        fields = defaultdict(list)
        for param in params:
            print(f'processing param: {param}')
            for t0 in [self.pdate - timedelta(hours=6), self.pdate]:
                if len(levelist)==0:
                    # surface variables
                    fsurf = self.era5_surf_download(t0)
                    grbs = pygrib.open(fsurf)
                    msgs = grbs.select(shortName=param)[self.member-1]
                    value = self.p050tp100(msgs.values,msgs.latlons()[0][:,0],msgs.latlons()[1][0,:],'linear')
                    value = value.flatten()
                    if np.isnan(value).any():  # if has NaN, filled with linearly interp
                        values = self.fill_nan_linear(value.copy())
                    else:
                        values = value
                    fields[msgs.shortName].append(values)
                    del fsurf, grbs, msgs, value, values
                else:
                    # pressure_level variables
                    fpl = self.era5_pl_download(t0)
                    grbs = pygrib.open(fpl)
                    for level in levelist:
                        msgs = grbs.select(shortName=param,level=level)[self.member-1]
                        value = self.p050tp100(msgs.values,msgs.latlons()[0][:,0],msgs.latlons()[1][0,:],'linear')
                        value = value.flatten()
                        if np.isnan(value).any():  # if has NaN, filled with linearly interp
                            values = self.fill_nan_linear(value.copy())
                        else:
                            values = value
                        vname = f"{param}_{level}" 
                        fields[vname].append(values)
                        del msgs,value,values,vname
                    del fpl,grbs
                del t0
        # Create a single matrix for each parameter
        for param, values in fields.items():
            fields[param] = np.stack(values)
        
        return fields
          
    def get_vars(self):
        fields = {}
        fields.update(self.get_data(params=self.param_sfc1))
        fields.update(self.get_data(params=self.para_pl1,levelist=self.pl_levels))
        # rename: surface z to orog (m2*s-2)
        orog = fields.pop('z')
        # orog = fields['z']
        fields['orog'] = orog; del orog
        # land mask to [0,1]
        fields['lsm'] = np.round(fields['lsm'])
        
        return fields

class Save2Zarr:
    def __init__(self,fields,fname,datetime,latitudes,longitudes):
        self.fields = fields
        self.fname = fname
        self.datetime = datetime
        self.ntime = len(datetime)
        self.nens = 1
        self.ncell = latitudes.shape[0]
        self.latitudes = latitudes
        self.longitudes = longitudes

    def SaveData(self):    
        variables = []; datas = [];
        for var_name, arr in self.fields.items():
            variables.append(var_name)
            datas.append(arr)
        variables = np.array(variables)
        datas = np.array(datas);
        name_to_index = {var: i for i, var in enumerate(variables)}
        data_new = np.expand_dims(np.transpose(datas, (1, 0, 2)), axis=2)  # [time,vars,ens,cell]
        del datas
        # Compute statistics
        mean = data_new.mean(axis=(0, 2, 3))
        minimum = data_new.min(axis=(0, 2, 3))
        squares = (data_new**2).mean(axis=(0, 2, 3))
        stdev = data_new.std(axis=(0, 2, 3))
        # convert coords
        ds = xr.Dataset(
            data_vars=dict(
                data=(["time", "variable", "ensemble", "cell"], data_new),
                mean=(["variable"], mean),
                minimum=(["variable"], minimum),
                squares=(["variable"], squares),
                stdev=(["variable"], stdev),
            ),
            coords=dict(
                time=("time", np.arange(self.ntime)),
                variable=("variable", variables),
                ensemble=("ensemble", np.arange(self.nens)),
                cell=("cell", np.arange(self.ncell)),
                latitudes=("cell", self.latitudes),
                longitudes=("cell", self.longitudes),
            ),
        )
        ds.attrs["name_to_index"] = name_to_index
        ds.to_zarr(self.fname, mode="w")
        nds = self.add_dates()
        nds.to_zarr(self.fname, mode="a")
        
    def add_dates(self) -> None:
        """Deal with the dates issue
        for some reason, it is a challenge to get the datetime64 dtype to open
        consistently between zarr and xarray, and
        it is much easier to deal with this all at once here
        than in the create_container and incrementally fill workflow.
        """
        xds = xr.open_zarr(self.fname)
        attrs = xds.attrs.copy()
        nds = xr.Dataset()
        nds["dates"] = xr.DataArray(
            self.datetime,
            coords=xds["time"].coords,
        )
        nds["dates"].encoding = {
            "dtype": "datetime64[s]",
            "units": "seconds since 1970-01-01",
        }
        nds.attrs = attrs
        return nds
        
#------------------------------------------------------
# python3 gene_ifs_era5_ics.py -s '20250801' -e '20250801' -k yes
# exec(open("gene_ifs_era5_ics.py").read())
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GEFS data for aifs_en ics")
    parser.add_argument("-s","--sdate", help="Processing start date in the format '%Y%m%d'")
    parser.add_argument("-e","--edate", help="Processing end date in the format '%Y%m%d'")
    parser.add_argument("-k", "--keep", help="Keep downloaded data (yes or no)", default="no")
    parser.add_argument("-H","--hour", help="Processing hour if processing singe time",type=int, required=False)
    args = parser.parse_args()
    sdate = args.sdate
    edate = args.edate
    keep_downloaded_data = args.keep.lower() == "yes"
    if args.hour is not None:
        hour = args.hour
        sdate = f"{sdate}_{hour:02d}"
        edate = f"{edate}_{hour:02d}"
    else:
        sdate = f"{sdate}_00"
        edate = f"{edate}_18"

    # reading the yaml file 
    with open("config_run.yaml", "r") as f:
        conf = yaml.safe_load(f)
    data_path = conf['data_path'] 
    dsource = conf['dsource'] 
    out_path = f'{data_path}/{dsource}'
    cache_path = f"{conf['cache_path']}/{dsource}"
    
    # lat/lon in zarr
    lat = np.arange(-90, 90.001, 1); lon = np.arange(0, 360, 1);
    lat2d,lon2d = np.meshgrid(lat, lon)
    latitudes = lat2d.ravel();   longitudes  = lon2d.ravel();

    # date arrays
    times = pd.date_range(
            start=pd.to_datetime(sdate, format="%Y%m%d_%H"),
            end=pd.to_datetime(edate, format="%Y%m%d_%H"),
            freq="6h"
            )
    print(times)
  
    for cdate in times:
        pdate = cdate.strftime('%Y-%m-%dT%H')
        out_dir = f"{out_path}/{pdate}"
        os.makedirs(out_dir, exist_ok=True)

        date_time = [cdate - timedelta(hours=6), cdate]
        # members/ only 10 members
        for member in np.arange(1,11):
            fname = f'{out_dir}/{dsource}_en_data_{pdate}_M{member}.zarr'
            print(f"Generating data: {fname}")
            if not os.path.exists(fname):
                data_processor = ERA5DataProcessor(cdate,member,cache_path)
                fields = data_processor.get_vars()  
                # save data as zarr
                data_save = Save2Zarr(fields,fname,date_time,latitudes,longitudes)
                data_save.SaveData()
            else:
                continue
        del member, pdate, out_dir



