"""
plot validations of ai-en-crps model: 
  --source: gfs/gefs_mean/ai_gefs/
  --reference: gdas/era5
"""

import yaml,argparse,os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.tri as tri
from collections import defaultdict


class DataProcessor:
    def __init__(self,fgdas,fgfs,fgefs,aigefs_path,lead_time,sdate,fhr,var,pl=None):
        self.fgdas = fgdas
        self.fgfs = fgfs
        self.fgefs = fgefs
        self.aigefs_path = aigefs_path
        self.lead_time = lead_time
        self.sdate = sdate
        self.fhr = fhr
        self.var = var
        self.pl  = pl
        self.MEMBERs = 31
    
    def gene_index(self):
        ds = xr.open_zarr(self.fgdas)
        val_time = ds.valid_time.values;
        dt_start = datetime.strptime(self.sdate, "%Y-%m-%dT%H")
        dt_end = dt_start + timedelta(hours=int(self.fhr))
        mask = val_time.astype('datetime64[ns]') == np.datetime64(dt_end)
        Tidx = np.where(mask)[0][0]
        if self.pl is not None:
            Pidx = np.where(ds.pressure.data==pl)[0][0]
        else:
            Pidx = []
        lat = ds.latitude.data;  lon = ds.longitude.data;
        return Tidx, Pidx, lat, lon 
    
    def read_gdas(self):
        Tidx, Pidx, lat, lon  = self.gene_index()
        ds = xr.open_zarr(self.fgdas)
        if self.pl is not None:
            value =  np.squeeze(ds[self.var].values[Tidx,:,Pidx,:,:])
        else:
            value = np.squeeze(ds[self.var].values[Tidx,:,:,:])
        return value, lon, lat
    
    def read_gfs(self):
        Tidx, Pidx, lat, lon  = self.gene_index()
        ds = xr.open_zarr(self.fgfs)
        if self.pl is not None:
            value =  np.squeeze(ds[self.var].values[:,Tidx,Pidx,:,:])
        else:
            value = np.squeeze(ds[self.var].values[:,Tidx,:,:])
        return value
        
    def read_gefs(self):
        Tidx, Pidx, lat, lon  = self.gene_index()
        ds = xr.open_zarr(self.fgefs)
        if self.pl is not None:
            value =  np.squeeze(ds[self.var].values[:,:,:,Pidx,:,:])
        else:
            value = np.squeeze(ds[self.var].values)
        return value
        
    def read_aigefs(self):
        Tidx, Pidx, lat, lon  = self.gene_index()
        vdata = []
        for MEMBER in np.arange(0,self.MEMBERs):
            file = f"{self.aigefs_path}/ai_gefs_en_{self.sdate}_{self.lead_time}h_M{MEMBER}.nc"
            ds = xr.open_dataset(file)
            if self.pl is not None:
                value = ds[f'{self.var}_{int(self.pl)}'].values[Tidx,:]
            else:
                value = ds[self.var].values[Tidx,:]
            vdata.append(value)
            del file, value
        vdata = np.array(vdata)
        vdata=vdata.reshape([self.MEMBERs,len(lat),len(lon)])
        
        return vdata

class DataStatics:
    def __init__(self,xvals,ytarget,lat,source,clim=None):
        self.xvals = xvals
        self.ytarget = ytarget
        self.source = source
        self.lat = lat
       
    def cal_statics(self):
        if self.source=='gefs':
            vals = np.nanmean(self.xvals,axis=0)
            vals_std = np.nanstd(self.xvals,axis=0, ddof=0)
            vals[np.isclose(self.lat, 90), :] = np.nan
            vals[np.isclose(self.lat, -90), :] = np.nan
            vals_std[np.isclose(self.lat, 90), :] = np.nan
            vals_std[np.isclose(self.lat, -90), :] = np.nan
        else:
            vals = self.xvals
        sta_vals = {}
        # Mean Squared Error (MSE)/# Root Mean Squared Error (RMSE)/# mean error/# Mean Absolute Error (MAE)
        mse= (vals - self.ytarget) ** 2
        sta_vals['ER'] = vals - self.ytarget
        sta_vals['AER'] = np.abs(vals - self.ytarget)
        sta_vals['RMSE'] = np.sqrt(np.nanmean(mse))
        sta_vals['MER'] = np.nanmean(sta_vals['ER'])
        sta_vals['MAER'] = np.nanmean(sta_vals['AER'])
        if self.source=='gefs':
            sta_vals['CRPS'] = self.crps_ensemble()
            sta_vals['SPRD'] = vals_std
        return sta_vals
    
    def crps_ensemble(self):
        """
        xens: ensemble forecasts, shape (n_members, ...), eg. [n_members,nx,ny]
        y: observed values, shape (...), eg. [nx,ny]
        """
        xens = self.xvals; y = self.ytarget;
        n = xens.shape[0]
        term1 = np.mean(np.abs(xens - y), axis=0)
        term2 = 0.5 * np.mean(np.abs(xens[:, None, ...] - xens[None, :, ...]), axis=(0, 1))
        value = term1 - term2
        value[np.isclose(self.lat, 90), :] = np.nan
        value[np.isclose(self.lat, -90), :] = np.nan
        return value

    def anomaly_correlation(self):
        """
        forecast, obs, clim: 2D arrays of same shape/ clim - mean values for a period
        """
        if self.source=='gefs':
            vals = np.nanmean(self.xvals,axis=0)
            vals[np.isclose(self.lat, 90), :] = np.nan
            vals[np.isclose(self.lat, -90), :] = np.nan
        else:
            vals = self.xvals   
        f_anom = vals - self.clim
        o_anom = self.ytarget - self.clim
        num = np.sum(f_anom * o_anom)
        den = np.sqrt(np.sum(f_anom**2) * np.sum(o_anom**2))
        return num / den
    
    def compute_cdf(self):
        """
        Compute forecast and observed CDFs for RPS/RPSS.
        
        Parameters
        ----------
        xens : np.ndarray
            Ensemble forecasts, shape (n_members, ny, nx)
        yobs : np.ndarray
            Observations, shape (ny, nx)
        bins : np.ndarray
            Thresholds defining the categories

        Returns
        -------
        forecast_cdf : np.ndarray
            Forecast cumulative probabilities, shape (n_bins, ny, nx)
        obs_cdf : np.ndarray
            Observation CDF (0 or 1), shape (n_bins, ny, nx)
        """
        xens = self.xvals; yobs = self.ytarget;
        
        n_bins = len(bins)
        n_members = xens.shape[0]

        forecast_cdf = np.zeros((n_bins, *xens.shape[1:]))
        obs_cdf = np.zeros((n_bins, *yobs.shape))

        for i, b in enumerate(bins):
            forecast_cdf[i] = np.mean(xens <= b, axis=0)
            obs_cdf[i] = (yobs <= b).astype(float)
            
        return forecast_cdf, obs_cdf

    def compute_ref_cdf(self):
        """
        Compute reference (climatology) CDF.
        
        Parameters
        ----------
        clim : np.ndarray
            Climatology samples, shape (n_samples, ny, nx)
        bins : np.ndarray
            Thresholds defining bins

        Returns
        -------
        ref_cdf : np.ndarray
            Reference cumulative probabilities, shape (n_bins, ny, nx)
        """
        n_bins = len(bins)
        ref_cdf = np.zeros((n_bins, *clim.shape[1:]))

        for i, b in enumerate(bins):
            ref_cdf[i] = np.mean(clim <= b, axis=0)
            
        return ref_cdf

    def rps(forecast_cdf, obs_cdf):
        """Ranked Probability Score"""
        return np.sum((forecast_cdf - obs_cdf)**2, axis=0)

    def rpss(self):
        """Ranked Probability Skill Score"""
        forecast_cdf, obs_cdf = self.compute_cdf()
        # Reference CDF from persistence: y_prev - observation values for the previous step, such as 6, then 6-6=0
        ref_cdf = (y_prev <= bins[:, None, None]).astype(float)
        
        rps_f = rps(forecast_cdf, obs_cdf)
        rps_ref = rps(ref_cdf, obs_cdf)
        return 1 - (rps_f / rps_ref)

class DataPlot:
    def __init__(self,xvals,var,conf,lon,lat,sdate,fh,source,plot_path,stype,pl=None):
        self.xvals = xvals
        self.var = var
        self.conf = conf
        self.lon = lon
        self.lat = lat
        self.sdate = sdate
        self.fh = fh
        self.source = source
        self.plot_path = plot_path
        self.pl = pl
        self.stype = stype
        self.lead_time = conf['lead_time']

        dt_start = datetime.strptime(self.sdate, "%Y-%m-%dT%H")
        dt_end = dt_start + timedelta(hours=int(self.fh))
        self.csdate = dt_end.strftime("%Y-%m-%dT%H")
    
    def get_var_name(self):
        if self.pl is not None:
            var_name = f'{self.var}_{int(self.pl)}'
        else:
            var_name = self.var
        return var_name
    
    def fix(self,lons,data):
        lon_new = np.where(lons > 180, lons - 360, lons)  # Shift the longitudes from 0-360 to -180-180
        sort_idx = np.argsort(lon_new)
        lon_sorted = lon_new[sort_idx]
        data_sorted = data[:, sort_idx]
        return lon_sorted, data_sorted
    
    def plot_values(self):
        var_name = self.get_var_name()
        vmin, vmax, nbin = self.conf[var_name][self.stype][0], self.conf[var_name][self.stype][1], self.conf[var_name][self.stype][2]
        levels = np.linspace(vmin, vmax, 50)
        cmap = self.conf['camp2']
        if self.stype=='VAL':
            if (self.source=='gefs') | (self.source=='aigefs'):
                xvals = np.nanmean(self.xvals,axis=0)
            else:
                xvals = self.xvals
        else:
            xvals = self.xvals[self.stype]
        lon, xvals = self.fix(self.lon,xvals)
        if (self.source=='gefs') | (self.source=='aigefs'):
            xvals[np.isclose(self.lat, 90), :] = np.nan
            xvals[np.isclose(self.lat, -90), :] = np.nan
        
        fig = plt.figure(figsize=(11, 6), layout='constrained')
        ax = plt.axes(projection = ccrs.PlateCarree())
        ax.set_extent([-180, 180, -90, 90])  # for global
        #ax.add_feature(cfeature.STATES, edgecolor='white', linewidth=2)
        ax.coastlines()
        im = ax.contourf(lon, self.lat, xvals, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='both')
        #define the xticks 
        ax.set_xticks(np.arange(-180, 180, 60), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        if (self.source=='gefs'):
            plt.title(f"GEFS_Ens-{var_name}_{self.stype} @ {self.sdate}_f{self.fh:03d}")
            ffig = f"{self.plot_path}/GEFS_Ens_{var_name}_{self.stype}_{self.sdate}_f{self.fh:03d}.png"
        elif (self.source=='aigefs'): 
            plt.title(f"AI_Ens-{var_name}_{self.stype} @ {self.sdate}_f{self.fh:03d}")
            ffig = f"{self.plot_path}/AI_Ens_{var_name}_{self.stype}_{self.sdate}_f{self.fh:03d}.png"
        elif (self.source=='gfs'): 
            plt.title(f"GFS-{var_name}_{self.stype} @ {self.sdate}_f{self.fh:03d}")
            ffig = f"{self.plot_path}/GFS_{var_name}_{self.stype}_{self.sdate}_f{self.fh:03d}.png"
        else: 
            plt.title(f"GDAS-{var_name}_{self.stype} @ {self.csdate}")
            ffig = f"{self.plot_path}/GDAS_{var_name}_{self.stype}_{self.csdate}.png"
        ax.set_yticks(np.arange(-90, 90, 30), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
        cb = plt.colorbar(im, ticks=np.linspace(vmin, vmax, nbin), shrink=0.6)
        cb.set_label(f'{self.stype} [{self.conf[var_name]['units']}]')
        plt.savefig(ffig, dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    def plot_times(self):
        var_name = self.get_var_name()
        fhrs = np.arange(0, self.lead_time+1, 6)
   
        if self.stype=='MER':
            da1_gefs = self.xvals['GEFS_MER']
            da1_aigefs = self.xvals['AIGEFS_MER']
            da2_gefs = self.xvals['GEFS_MAER']
            da2_aigefs = self.xvals['AIGEFS_MAER']
            ylabs = "MERR(solid) and ABS. ERR(dash)"
            ffig = f"{self.plot_path}/GEFS_Ens_{var_name}_MER_MAER_{self.sdate}.png"
        if self.stype=='RMSE':
            da1_gefs = self.xvals['GEFS_RMSE']
            da1_aigefs = self.xvals['AIGEFS_RMSE']
            da2_gefs = self.xvals['GEFS_SPRD']
            da2_aigefs = self.xvals['AIGEFS_SPRD']
            ylabs = "RMSE(solid) and SPREAD(dash)"
            ffig = f"{self.plot_path}/GEFS_Ens_{var_name}_RMSE_SPREAD_{self.sdate}.png"
        if self.stype=='CRPS':
            da1_gefs = self.xvals['GEFS_CRPS']
            da1_aigefs = self.xvals['AIGEFS_CRPS']
            ylabs = "CRPS"
            ffig = f"{self.plot_path}/GEFS_Ens_{var_name}_CRPS_{self.sdate}.png"

        labs = ['GEFSV12','AI_GEFS']
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(fhrs, da1_gefs, color="blue", linewidth=2.0)
        plt.plot(fhrs, da1_aigefs, color="red", linewidth=2.0)
        if (self.stype=='MER') | (self.stype=='RMSE'):
            plt.plot(fhrs, da2_gefs, color="blue", linestyle="--", linewidth=1.5)
            plt.plot(fhrs, da2_aigefs, color="red", linestyle="--", linewidth=1.5)
        if self.stype=='MER':
            plt.title(f"Global {var_name} \nEnsemble Mean Error and Ensemble Abs. Error\nAverage For {self.sdate}", fontsize=12, fontweight="bold")
        if self.stype=='RMSE':
            plt.title(f"Global {var_name} \nEnsemble Mean RMSE and Ensemble SPREAD\nAverage For {self.sdate}",fontsize=12, fontweight="bold")
        if  self.stype=='CRPS':
            plt.title(f"Global {var_name} \nContinuous Ranked Probability Score\nAverage For {self.sdate}",fontsize=12, fontweight="bold")
        plt.xlabel("Forecast hours", fontsize=12)
        plt.ylabel(ylabs, fontsize=12)
        plt.xticks(np.arange(0, 241, 24))
        plt.grid(True, linestyle=":", linewidth=0.7)
        if self.stype=='MER':
            plt.ylim(-2, 4)
        if self.stype=='RMSE':
            plt.ylim(0, 5)
        if self.stype=='CRPS':
            plt.ylim(0, 5)
       
        if (var_name=='u_250') | (var_name=='v_250'):
            if self.stype=='MER':
                plt.ylim(-2, 12)
            if self.stype=='RMSE':
                plt.ylim(0, 15)
            if self.stype=='CRPS':
                plt.ylim(0, 10)
        if (var_name=='u_850') | (var_name=='v_850'):
            if self.stype=='MER':
                plt.ylim(-2, 6)
            if self.stype=='RMSE':
                plt.ylim(0, 8)

        plt.xlim(0, 240)
        plt.legend(labs, loc="upper left", fontsize=10)
        plt.tick_params(labelsize=11, direction="in", length=6, width=1.2)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.2)
    
        plt.tight_layout()
        plt.savefig(ffig, dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close()
              

#-----------------------------------------------------------------

# python3 plot_validation.py --sdate 2024-02-20T00 --var '2t' --pres      
#reading the start date
parser = argparse.ArgumentParser(description="Data processing for sdate")
parser.add_argument("--sdate", type=str, required=True, help="Process start date in format yyyy-mm-ddThh")
parser.add_argument("--var", type=str, required=True, help="variables")
parser.add_argument("--pres", type=float, required=False, help="pressure level (in hPa)")
args = parser.parse_args()
sdate = args.sdate
var = args.var
pl = args.pres if args.pres is not None else None

# reading the yaml file 
with open("jobcard_config.yaml", "r") as f:
    conf = yaml.safe_load(f)
lead_time = conf['lead_time']
data_path = conf['data_path']
pred_path = conf['pred_path']
plot_path = conf['plot_path']

# calculate the end date based on the start date and lead_time
dt_start = datetime.strptime(sdate, "%Y-%m-%dT%H")
dt_end = dt_start + timedelta(hours=lead_time)
edate = dt_end.strftime("%Y-%m-%dT%H")
del dt_start,dt_end

# the file of each data source
gdas_path = f"{data_path}/gdas/gdas_{sdate}_{edate}.zarr"
gfs_path = f"{data_path}/gfs/gfs_{sdate}_f{lead_time}.zarr"
gefs_path = f"{data_path}/gefs/{sdate}"
aigefs_path = f"{pred_path}/{sdate}"
plot_path = f"{plot_path}/{sdate}"
os.makedirs(plot_path, exist_ok=True)

# process each fh for 0 to lead_time
Rsta_lt = defaultdict(list)
for fh in np.arange(0,lead_time+1,6):
    fgefs = f"{gefs_path}/gefs_{sdate}_f{fh:03d}.zarr"
    print(f'processing the lead time @ {fh:03d}h')
    
    # processing data
    data_processor = DataProcessor(gdas_path,gfs_path,fgefs,aigefs_path,lead_time,sdate,fh,var,pl)
    gdas_val,lons,lats = data_processor.read_gdas()
    gfs_val = data_processor.read_gfs()
    gefs_val = data_processor.read_gefs()
    aigefs_val = data_processor.read_aigefs()
    
    # calculating statics
    process_statics = DataStatics(gfs_val,gdas_val,lats,'gfs',clim=None)
    gfs_stas = process_statics.cal_statics();
    Rsta_lt['GFS_MER'].append(gfs_stas['MER'])
    Rsta_lt['GFS_MAER'].append(gfs_stas['MAER'])
    Rsta_lt['GFS_RMSE'].append(gfs_stas['RMSE'])
    del process_statics;

    process_statics = DataStatics(gefs_val,gdas_val,lats,'gefs',clim=None)
    gefs_stas = process_statics.cal_statics()
    Rsta_lt['GEFS_MER'].append(gefs_stas['MER'])
    Rsta_lt['GEFS_MAER'].append(gefs_stas['MAER'])
    Rsta_lt['GEFS_RMSE'].append(gefs_stas['RMSE'])
    Rsta_lt['GEFS_CRPS'].append(np.nanmean(gefs_stas['CRPS']))
    Rsta_lt['GEFS_SPRD'].append(np.nanmean(gefs_stas['SPRD']))
    del process_statics;
 
    process_statics = DataStatics(aigefs_val,gdas_val,lats,'gefs',clim=None)
    aigefs_stas = process_statics.cal_statics()
    Rsta_lt['AIGEFS_MER'].append(aigefs_stas['MER'])
    Rsta_lt['AIGEFS_MAER'].append(aigefs_stas['MAER'])
    Rsta_lt['AIGEFS_RMSE'].append(aigefs_stas['RMSE'])
    Rsta_lt['AIGEFS_CRPS'].append(np.nanmean(aigefs_stas['CRPS']))
    Rsta_lt['AIGEFS_SPRD'].append(np.nanmean(aigefs_stas['SPRD']))
    del process_statics;
    
    # plot results
    #   -- values: 2d
    # process_plot = DataPlot(gdas_val,var,conf,lons,lats,sdate,fh,'gdas',plot_path,'VAL',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(gfs_val,var,conf,lons,lats,sdate,fh,'gfs',plot_path,'VAL',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(gefs_val,var,conf,lons,lats,sdate,fh,'gefs',plot_path,'VAL',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(aigefs_val,var,conf,lons,lats,sdate,fh,'aigefs',plot_path,'VAL',pl); process_plot.plot_values(); del process_plot
    
    # #   -- statics: bias, absoult bias, rmse (2d)
    # process_plot = DataPlot(gfs_stas,var,conf,lons,lats,sdate,fh,'gfs',plot_path,'ER',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(gefs_stas,var,conf,lons,lats,sdate,fh,'gefs',plot_path,'ER',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(aigefs_stas,var,conf,lons,lats,sdate,fh,'aigefs',plot_path,'ER',pl); process_plot.plot_values(); del process_plot
    
    # process_plot = DataPlot(gfs_stas,var,conf,lons,lats,sdate,fh,'gfs',plot_path,'AER',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(gefs_stas,var,conf,lons,lats,sdate,fh,'gefs',plot_path,'AER',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(aigefs_stas,var,conf,lons,lats,sdate,fh,'aigefs',plot_path,'AER',pl); process_plot.plot_values(); del process_plot

    # process_plot = DataPlot(gefs_stas,var,conf,lons,lats,sdate,fh,'gefs',plot_path,'CRPS',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(aigefs_stas,var,conf,lons,lats,sdate,fh,'aigefs',plot_path,'CRPS',pl); process_plot.plot_values(); del process_plot 
    
    # process_plot = DataPlot(gefs_stas,var,conf,lons,lats,sdate,fh,'gefs',plot_path,'SPRD',pl); process_plot.plot_values(); del process_plot
    # process_plot = DataPlot(aigefs_stas,var,conf,lons,lats,sdate,fh,'aigefs',plot_path,'SPRD',pl); process_plot.plot_values(); del process_plot    


#print(np.max(Rsta_lt['GEFS_MER']))
#print(np.max(Rsta_lt['GEFS_MAER']))
#print(np.max(Rsta_lt['GEFS_RMSE']))
#print(np.max(Rsta_lt['GEFS_CRPS']))

# plot region mean of mer/maer/rmse/spread/crps for ensemble forecasts
#process_plot = DataPlot(Rsta_lt,var,conf,lons,lats,sdate,fh,'gefs',plot_path,'MER',pl); process_plot.plot_times(); del process_plot 
#process_plot = DataPlot(Rsta_lt,var,conf,lons,lats,sdate,fh,'gefs',plot_path,'RMSE',pl); process_plot.plot_times(); del process_plot
process_plot = DataPlot(Rsta_lt,var,conf,lons,lats,sdate,fh,'gefs',plot_path,'CRPS',pl); process_plot.plot_times(); del process_plot




