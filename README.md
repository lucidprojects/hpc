# NYU GREENE HPC
UPDATED for NYU's new GREENE HPC.  forked from the awesome guide [Cvalenzuela](https://github.com/cvalenzuela/hpc) put together. 

A quick reference to access NYU's High Performance Computing Greene Cluster.

The official wiki is [here](https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/greene), this is an unofficial document created as a quick-start guide for first-time users with a focus in Python.

## Get an account

You need to be affiliated to NYU and have a sponsor. 

To get an account approved, ask a professor for sponsorship and complete this [form.](https://docs.google.com/forms/d/e/1FAIpQLSejXQemyQij389ipNdhFCQ-PD8AjSa_B6mhdQudB6DCdPejbw/viewform) It requires their name and/or NetID

## Log in

Once you have been approved, you can access HPC from:

 1. Within the NYU network:

```bash
ssh NYUNetID@2@greene.hpc.nyu.edu
```

Once logged in, the root should be:
`/home/NYUNetID`, so running `pwd` should print:

```bash
[NYUNetID@log-0 ~]$ pwd
/home/NYUNetID
```

2. From an off-campus location:

First, login to the bastion host:

```bash
ssh NYUNetID@gw.hpc.nyu.edu
```

Then login to the cluster:

```bash
ssh greene.hpc.nyu.edu
```

Or use NYU VPN:
Install config instructions [here](http://www.nyu.edu/life/information-technology/getting-started/network-and-connectivity/vpn.html)

Then
```bash
ssh NYUNetID@greene.hpc.nyu.edu
```

## File Systems

You can get acces to three filesystems: `/home`, `/scratch`, and `/archive`.

Scratch is a file system mounted on Greene that is connected to the compute nodes where we can upload files faster. Notice that the content gets periodically flushed.

```bash
[NYUNetID@log-0 ~]$ cd /scratch/NYUNetID
[NYUNetID@log-0 ~]$ pwd
/scratch/NYUNetID
```

`/home` and `/scratch` are separate filesystems in separate places, but you should use `/scratch` to store your files.

Greene HPC setup notes

## Configure [Singularity](https://sylabs.io/guides/3.7/user-guide/) Container and environment

This process is the replacement for loading modules that was used on Prince.

Setup container dir

```bash
$mkdir [DIRNAME]
$cd [DIRNAME]
```

Copy environment overlay file to [DIRNAME] and gunzip it
```bash
$cp -rp /scratch/work/public/overlay-fs-ext3/overlay-5GB-200K.ext3.gz .
$gunzip overlay-5GB-200K.ext3.gz
```
Now run the overlay with the proper singularity environment container.

Check which singularity container versions are available 
```bash
$ls /scratch/work/public/singularity/
```

For PyTorch - per this [link](https://pytorch.org)
the lastest version 1.8.1 will work with Cuda 10.2 or 11.1

*Note Derrick's colab installs these versions
```bash
Name: torch
Version: 1.8.0+cu101
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /usr/local/lib/python3.7/dist-packages
Requires: numpy, typing-extensions
Required-by: torchvision, torchtext, fastai
Python 3.7.10
```

Start up container
```bash
$singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
```

Install miniconda and set bash path
```bash
$wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
$export PATH=/ext3/miniconda3/bin:$PATH
$conda update -n base conda -y
```
create a wrapper script /ext3/env.sh 
```bash
$nano /ext3/env.sh 
```

Paste in
```bash
#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
```

Exit and restart the container
```bash
$singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
$source /ext3/env.sh 
```

Check versions and paths

```bash
$which python
$which pip
$which conda
$python --version
```

***NOTE Derrick's PyTorch Colab uses Python 3.7.10 
PyTorch site says it will work with 3.5 or greater. https://pytorch.org/get-started/locally/  The above sif file installs 3.8.5

Install PyTorch and dependencies
```bash
$pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

*Note Derrick's Colab also installs these
```bash
$pip install ninja opensimplex
```

We also need tensorboard which is not included above
```bash
conda install -c conda-forge tensorboard 
```

check version
```bash
$pip show torch
```

create and run torch_test.py

```bash
$nano torch_test.py
```

Paste
```bash
import torch
x = torch.rand(5, 3)
print(x)
```

Should output a tensor similar to 
```bash
tensor([[0.3216, 0.3976, 0.0339],
        [0.4608, 0.2480, 0.8459],
        [0.2098, 0.6496, 0.6744],
        [0.3855, 0.8929, 0.6023],
        [0.7854, 0.2878, 0.5161]])
``` 
If that doesn't work there may be some versioning / dependency issues

Check pip versions installed with 
```bash
$pip list
```

If torch_test.py works clean up
```bash
$conda clean --all --yes
```

Exit the container and rename to something you'll remember

```bash
$exit
$mv overlay-5GB-200K.ext3 pytorch1.8.0-cuda11.0.ext3
```

It is also helpful to create a bash to launch the singularity container
```bash
$nano run-singularity.bash
```

Paste

```bash
#!/bin/bash

module purge

#check hostname and make sure "-nv" arg is added (-n = run container in a new network namespace, -v = verbose)
if [[ "$(hostname -s)" =~ ^g[r,v] ]]; then nv="--nv"; fi

singularity exec $nv \
	    --overlay /home/[NetID]/[DIRNAME]/pytorch1.8.0-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash
```

You can also use this as the base for you actual SBATCH bash file

If all the above checks out create SBATCH file
Adjust cpu, mem, gpu, time, job-name, mail-user, output accordingly
Adjust train.py args as desired

*note make sure all directories listed below exist.
*note :ro after overlay makes it "read only" so we can still open the singularity container and use it for other jobs. :rw (default) would lock the file for use

```bash
$nano [jobname].sbatch
```

```bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=[jobname]
#SBATCH --mail-type=END
#SBATCH --mail-user=[NetID]@nyu.edu
#SBATCH --output=/scratch/[NetID]/output/[jobname]_%j.out

module purge

cd /home/[NetID]/stylegan2-ada-pytorch/

singularity exec --nv \
	--overlay /home/[NetID]/[DIRNAME]/pytorch1.8.0-cuda11.0.ext3:ro \
        /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
        bash -c "source /ext3/env.sh; \
        python train.py --gpus=4 --cfg=11gb-gpu --metrics=None --outdir=/scratch/[NetID]/results \
        --data=/scratch/[NetID]/datasets/[dataset.zip] --snap=4 --resume=/scratch/[NetID]/pretrained/[pretrianed or resume].pkl \
        --augpipe=bg --initstrength=0.0 --gamma=50.0 \
        --mirror=True --mirrory=False --nkimg=0"
        
```
        
Transfer dataset and files to Greene

Rclone with module rclone/1.53.3

```bash
$ module avail
$ module load rclone/1.53.3  
```
[Rclone documentation](https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/data-management/transfering-data/google-drive
)

or

SCP 
to Greene
```bash
scp [pretrianed].pkl [NetID]@gdtn.hpc.nyu.edu:/scratch/[NetID]/pretrained/ .
scp [dataset].zip [NetID]@gdtn.hpc.nyu.edu:/scratch/[NetID]/datasets/ .
```

from Greene
```bash
scp [NetID]@gdtn.hpc.nyu.edu:/scratch/[NetID]/[PATH_TTO_FILE] .
```

## Helpful cmds

Show your virtual env disk usage.  Make sure you keep /home/[NetID]/  relatively pruned
```bash
$myquota 
```

Submit job
```bash
$sbatch [jobname].sbatch
```
Check your job in the queue
```bash
$squeue -u [NetID]
```
more details
```bash
squeue -j [JOB_ID] -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R %m"
             JOBID PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON) MIN_MEMORY
           3687633 rtx8000,v [jobname]   [NetID]  PENDING       0:00 2-00:00:00      1 (Resources) 200G
```           
See full sbatch queue
```bash
$sprio
```


## Loading Modules

### This portion has mostly been replaced by the Singularity setup above.  Leaving it incase there is a usecase for modules, e.g. rclone mentioned above.

Slurm allows you to load and manage multiple versions and configurations of software packages.

To see available package environments:
```bash
module avail
```

To load a model:
```bash
module load [package name]
```

For example if you want to use Tensorflow-gpu:
```bash
module load cudnn/8.0v6.0
module load cuda/8.0.44
module load tensorflow/python3.6/1.3.0
```

To check what is currently loaded:
```bash
module list
```

To remove all packages:
```bash
module purge
```

