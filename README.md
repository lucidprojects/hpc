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

Scratch is a file system mounted on Prince that is connected to the compute nodes where we can upload files faster. Notice that the content gets periodically flushed.

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
$nano [my_job].sbatch
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

rclone 
rclone with module rclone/1.53.3

```bash
$ module avail
$ module load rclone/1.53.3  
```
[Rclone documentation](https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/data-management/transfering-data/google-drive
)

or

SCP
```bash
scp [pretrianed].pkl [NetID]@gdtn.hpc.nyu.edu:/scratch/[NetID]/pretrained/
scp [dataset].pkl [NetID]@gdtn.hpc.nyu.edu:/scratch/[NetID]/datasets/
```

Helpful cmds

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
           3687633 rtx8000,v desert10   [NetID]  PENDING       0:00 2-00:00:00      1 (Resources) 200G
```           
See full sbatch queue
```bash
$sprio
```


## Loading Modules

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

To get helpful information about the package:
```bash
module show torch/gnu/20170504
```
Will print something like 
```bash
--------------------------------------------------------------------------------------------------------------------------------------------------
   /share/apps/modulefiles/torch/gnu/20170504.lua:
--------------------------------------------------------------------------------------------------------------------------------------------------
whatis("Torch: a scientific computing framework with wide support for machine learning algorithms that puts GPUs first")
whatis("Name: torch version: 20170504 compilers: gnu")
load("cmake/intel/3.7.1")
load("cuda/8.0.44")
load("cudnn/8.0v5.1")
load("magma/intel/2.2.0")
...
```
`load(...)` are the dependencies that are also loaded when you load a package.


## Interactive Mode: Request CPU

You can submit batch jobs in prince to schedule jobs. This requires to write custom bash scripts. Batch jobs are great for longer jobs, and you can also run in interactive mode, which is great for short jobs and troubleshooting.

To run in interactive mode:

```bash 
[NYUNetID@log-0 ~]$ srun --pty /bin/bash
```

This will run the default mode: a single CPU core and 2GB memory for 1 hour.

To request more CPU's:

```bash
[NYUNetID@log-0 ~]$ srun -n4 -t2:00:00 --mem=4000 --pty /bin/bash
[NYUNetID@c26-16 ~]$ 
```
That will request 4 compute nodes for 2 hours with 4 Gb of memory.


To exit a request:
```
[NYUNetID@c26-16 ~]$ exit
[NYUNetID@log-0 ~]$
```

## Interactive Mode: Request GPU

```bash
[NYUNetID@log-0 ~]$ srun --gres=gpu:1 --pty /bin/bash
[NYUNetID@gpu-25 ~]$ nvidia-smi
Mon Oct 23 17:49:19 2017
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 0000:12:00.0     Off |                    0 |
| N/A   37C    P8    29W / 149W |      0MiB / 11439MiB |      0%   E. Process |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```


## Submit a job

You can write a script that will be executed when the resources you requested became available.

A simple CPU demo:

```bash
## 1) Job settings

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=CPUDemo
#SBATCH --mail-type=END
#SBATCH --mail-user=itp@nyu.edu
#SBATCH --output=slurm_%j.out
  
## 2) Everything from here on is going to run:

cd /scratch/NYUNetID/demos
python demo.py
```

Request GPU:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mem=3GB
#SBATCH --job-name=GPUDemo
#SBATCH --mail-type=END
#SBATCH --mail-user=itp@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/NYUNetID/trainSomething
source activate ML
python train.py
```

Submit your job with:

```bash
sbatch myscript.s
```

Monitor the job:

```bash
squeue -u $USER
```

More info [here](https://wikis.nyu.edu/display/NYUHPC/Submitting+jobs+with+sbatch)

## Setting up a tunnel

To copy data between your workstation and the NYU HPC clusters, you must set up and start an SSH tunnel.

What is a tunnel?

> "A tunnel is a mechanism used to ship a foreign protocol across a network that normally wouldn't support it."<sup>[1](http://www.enterprisenetworkingplanet.com/netsp/article.php/3624566/Networking-101-Understanding-Tunneling.htm)</sup>

1. In your local computer root directory, and if you don't have it already, create a folder called `/.shh`:
```bash
mkdir ~/.ssh
```

2. Set the permission to that folder:
```bash
chmod 700 ~/.ssh
```

3. Inside that folder create a new file called `config`:
```bash
touch config
```

4. Open that file in any text editor and add this: 
```bash
# first we create the tunnel, with instructions to pass incoming
# packets on ports 8024, 8025 and 8026 through it and to specific
# locations
Host hpcgwtunnel
   HostName gw.hpc.nyu.edu
   ForwardX11 no
   LocalForward 8025 dumbo.hpc.nyu.edu:22
   LocalForward 8026 prince.hpc.nyu.edu:22
   User NetID 
# next we create an alias for incoming packets on the port. The
# alias corresponds to where the tunnel forwards these packets
Host dumbo
  HostName localhost
  Port 8025
  ForwardX11 yes
  User NetID

Host prince
  HostName localhost
  Port 8026
  ForwardX11 yes
  User NetID
```

Be sure to replace the `NetID` for your NYU NetId

## Transfer Files

To copy data between your workstation and the NYU HPC clusters, you must set up and start an SSH tunnel. (See previous step)


1. Create a tunnel
```bash
ssh hpcgwtunnel
```
Once executed you'll see something like this:

```bash
Last login: Wed Nov  8 12:15:48 2017 from 74.65.201.238
cv965@hpc-bastion1~>$
```

This will use the settings in `/.ssh/config` to create a tunnel. **You need to leave this open when transfering files**. Leave this terminal tab open and open a new tab to continue the process.

2. Transfer files

### Between your computer and the HPC

- A File:
```bash
scp /Users/local/data.txt NYUNetID@prince:/scratch/NYUNetID/path/
```

- A Folder:
```bash
scp -r /Users/local/path NYUNetID@prince:/scratch/NYUNetID/path/
```

### Between the HPC and your computer

- A File:
```bash
scp NYUNetID@prince:/scratch/NYUNetID/path/data.txt /Users/local/path/
```

- A Folder:
```bash
scp -r NYUNetID@prince:/scratch/NYUNetID/path/data.txt /Users/local/path/ 
```

## Screen

Create a `./.screenrc` file and append this [gist](https://gist.github.com/joaopizani/2718397)
