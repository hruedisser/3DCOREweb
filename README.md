# 3DCOREweb

Reconstruct CMEs using the 3D Coronal Rope Ejection Model (adapted from [A.J. Weiss](https://github.com/ajefweiss/py3DCORE)).

## Installation
------------
First install new conda environment:

    conda create -n "3dcorenv" python=3.10.10
    conda activate 3dcorenv
    
Install the latest version of HelioSat manually using `git`: (Optional - to avoid using heliosat, download [data archive](https://doi.org/10.6084/m9.figshare.11973693.v23) at and place the files in 3DCOREweb/src/coreweb/dashcore/data/archive)

    git clone https://github.com/ajefweiss/HelioSat
    cd HelioSat
    pip install -e .
    
Install the latest version manually using `git`:

    git clone https://github.com/hruedisser/3DCOREweb
    cd py3DCORE
    pip install -e .
    
Install all necessary packages:
    
    pip install -r requirements.txt
    

------------

## 3DCOREweb application
------------

To start the application:

    3DCOREweb start

By default, no event is selected. The user is asked to continue with one of three options to select an event, that will be processed. To speed up data loading during the usage of the app, the user can choose to download the Data Archive using the according button.

![image](https://github.com/hruedisser/3DCOREweb/assets/75985139/d11e5fe7-52c4-4b97-a5dc-d26ad469e205)

Using the Catalog option, several filters can be applied to search the list of available events.

![image](https://github.com/hruedisser/3DCOREweb/assets/75985139/8f3ccacf-11e0-4047-94c4-6921835d2ea9)

Clicking the Submit button will start the data preprocessing.

![image](https://github.com/hruedisser/3DCOREweb/assets/75985139/f6b1053f-4948-411d-997b-5a1215153c21)

The loading symbol indicates that the data preprocessing is still in progress.

![image](https://github.com/hruedisser/3DCOREweb/assets/75985139/97fd805b-09c4-43b9-9ed8-42245b1cb1c2)

Once it is done, the loading symbol is replaced by a checkmark and the user can continue to the next step.

![image](https://github.com/hruedisser/3DCOREweb/assets/75985139/26501687-ceb0-471d-8b15-b3734f636252)

On the Plot subpage, a 2D plot showing the spacecraft and planet positions around the time of the event can be found. The user can use the checkboxes to adjust which components should be visible in the plot.

![image](https://github.com/hruedisser/3DCOREweb/assets/75985139/d407e3ff-e500-4904-8619-250d4c60da35)

On the bottom of the page the insitu data measured by the previously chosen spacecraft can be seen. The colored background indicates the boundaries of the cataloged event.

![image](https://github.com/hruedisser/3DCOREweb/assets/75985139/5208d60d-c780-4d21-9c17-1315ebcc1a03)

Adding a synthetic spacecraft, the user can adjust its position and reload the plot.

![image](https://github.com/hruedisser/3DCOREweb/assets/75985139/f7e94a98-da7e-4b26-9c59-c6190c22b4e5)



    
------------
## Notes on HelioSat
------------

3DCORE uses the package [HelioSat](https://github.com/ajefweiss/HelioSat) to retrieve spacecraft data and other spacecraft related information if the data is not made available via the [archive](https://doi.org/10.6084/m9.figshare.11973693.v23) (positions, trajectories, etc). 

In order for HelioSat to work properly, the following steps are necessary:

1. manually create the folder ~/.heliosat 
2. within .heliosat, manually create the following three folders
    - cache
    - data
    - kernels
3. if HelioSat fails to download kernels, download them manually and place them in the kernel folder

In those folders, HelioSat will download and save the needed spacecraft data and corresponding kernels. 
If you want to use custom data not available online, place the datafile in .heliosat/data and set custom_data = True during fitting.
