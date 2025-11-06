import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import yaml
import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ
import subprocess,time

def write_yaml(
    checkpoint,
    lead_time,
    DATE,
    dfile,
    pfile,
    output_variables=None,
    ):
    """
    Generate an Anemoi inference YAML config file in exact style.
    """
    patch_metadata = CommentedMap()
    patch_metadata["dataset"] = CommentedMap()
    constant_fields = CommentedSeq(["lsm", "orog"])
    constant_fields.fa.set_flow_style()  # force inline: [lsm, orog]
    patch_metadata["dataset"]["constant_fields"] = constant_fields
    # Use CommentedMap to add comments and preserve order
    data = CommentedMap()
    data["checkpoint"] = DQ(checkpoint)
    data["lead_time"] = lead_time
    data.yaml_add_eol_comment("hours", "lead_time")  # comment after lead_time
    data["date"] = DATE
    # input/output
    data["input"] = CommentedMap()
    data["input"]["dataset"] = DQ(dfile)
    #data["output"] = "printer"
    if output_variables==None:
        data["output"] = CommentedMap()
        data["output"]["netcdf"] = DQ(pfile)
    else:
        data["output"] = CommentedMap()
        data["output"]["netcdf"] = CommentedMap()
        data["output"]["netcdf"]["path"] = DQ(pfile)
        cf_vars = CommentedSeq(output_variables)
        cf_vars.fa.set_flow_style()  # inline list
        data["output"]["netcdf"]["variables"] = cf_vars
    
    data["write_initial_state"] = True
    # patch_metadata
    data["patch_metadata"] = patch_metadata
    return data
    
#----------------------
# usage: # python3 jobcard_submit_model.py --sdate 2025-08-01T00 --edate 2025-08-01T00 
#----------------------

# date for processing
parser = argparse.ArgumentParser(description="Data processing from sdate to edate")
parser.add_argument("--sdate", type=str, required=True, help="Process start date in format yyyy-mm-ddThh")
parser.add_argument("--edate", type=str, required=True, help="Process end date in format yyyy-mm-ddThh")
args = parser.parse_args()
sdate = args.sdate
edate = args.edate

# config
with open("config_run.yaml", "r") as f:
    conf = yaml.safe_load(f)   
conf["pred_path"] = conf["pred_path"].replace("${dsource}", conf["dsource"])

if conf['dsource']=='era5':
    data_path = f"{conf['data_path']}/era5"
else:
    data_path = f"{conf['data_path']}/Gefs4Era5Crps"
lead_time = conf['lead_time']
job_path = conf['fjob']
os.makedirs(job_path,exist_ok=True)

if conf['mICs']:
    data_path = f"{data_path}/{conf['mic_str']}"

# Template of the job card
job_card_template = """#!/bin/bash
#SBATCH -A gpu-ai4wp #nems
#SBATCH -J inference_{dsource}_{pt}_M{MEMBER}
#SBATCH -o logs/inference_{dsource}_{pt}_M{MEMBER}.out
#SBATCH -e logs/inference_{dsource}_{pt}_M{MEMBER}.err
#SBATCH --nodes=1
#SBATCH -t 00:10:00 #2-00:00:00
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu #gpuwf
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G 
#SBATCH --ntasks-per-node=1

# module load and activate en
source /scratch3/NCEPDEV/nems/Jianjun.Liu/miniconda/etc/profile.d/conda.sh
conda activate anemoi

anemoi-inference run {job_path}/inference_config_{dsource}_{pt}_M{MEMBER}.yaml
# srun anemoi-inference run {job_path}/inference_config_{dsource}_{pt}_M{MEMBER}.yaml
"""
 
# Generate and submit job cards
times = pd.date_range(start=sdate, end=edate, freq="6h")
for pdate in times:
    pt = pdate.strftime('%Y-%m-%dT%H')
    pred_path = f"{conf['pred_path']}/{pt}"
    os.makedirs(pred_path,exist_ok=True)
    
    job_ids = []
    for MEMBER in np.arange(conf['NM'][0],conf['NM'][1]):
        pfile = f"{pred_path}/ai_{conf['dsource']}_en_{pt}_{lead_time}h_epoch{conf['epoch']}_M{MEMBER}.nc"
        if os.path.exists(pfile):
            continue

        # write a slurm
        job_card = job_card_template.format(pt=pt,MEMBER=MEMBER,job_path=job_path,dsource=conf['dsource'])
        job_filename = f"{job_path}/submit_inference_{pt}_M{MEMBER}.sh"
        with open(job_filename, 'w') as job_file:
            job_file.write(job_card)
        del job_card

        # write the inference_config.yaml file
        if conf['mICs']:
            dfile = f"{data_path}/{conf['dsource']}_en_data_{conf['mic_str']}_M{MEMBER}.zarr"
        else:
            dfile = f"{data_path}/{pt}/{conf['dsource']}_en_data_{pt}_M{MEMBER}.zarr"
        # dfile = f"{data_path}/{pt}/{conf['dsource']}_en_data_{pt}.zarr"  # era5 reanalysis data
            
        data = write_yaml(conf['checkpoint'],lead_time,pt,dfile,pfile,conf['output_variables'])
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.preserve_quotes = True
        yaml_filename = f"{job_path}/inference_config_{conf['dsource']}_{pt}_M{MEMBER}.yaml"
        with open(yaml_filename, "w") as f:
            yaml.dump(data, f)
        del dfile,pfile,data,yaml_filename
        
        # Submit the job and capture job ID
        result = subprocess.run(["sbatch", job_filename], capture_output=True, text=True)
        print(result.stdout.strip())
        try:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
        except Exception:
            print(f"⚠️ Could not extract job ID for time:{pt} @ M{MEMBER}")
            
    # === Wait for all member jobs of this date to finish ===
    print(f"🕒 Waiting for all jobs for {pt} to finish...")  
    while True:
        active = False
        for job_id in job_ids:
            check = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
            if job_id in check.stdout:
                active = True
                break
        if not active:
            print(f"✅ All jobs for {pt} completed.")
            break
        time.sleep(60)
print("\n🎯 All dates processed successfully.")  
