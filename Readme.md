### Sequential Stacking Link Prediction Algorithms for Temporal Networks 

This is the github repo accompany the paper by Xie He, Amir Ghasemian, Eun Lee, Aaron Clauset, Peter Mucha. 
The paper is currently under revision at Nature Communications.

All copyright reserved. 

**Please cite the paper when using the data or code.**

### System requirements

Use the commend

```bash
conda env create -f environment.yml
```

to create the required environmnet for the code.
To reproduce our experiments, you will need at least Python 3.10 and a few packages installed(see the enviornment file for specific details). 

You can check your python version with

```bash
$ python --version
```

### To run only the Top-Sequential Experiments

Running the experiments will take a while depending on your hardware.
The best way to run only the Top-Sequential Experiment is to follow the $example.py$ file. 

```bash
$ cd experiments
$ python run_erdos_renyi.py
$ python plot_erdos_renyi.py
```
Change the variables in run_erdos_renyi.py to run with different settings i.e. number of layers and whether isomorphism counting is being done.

plot_erdos_renyi.py will generate a figure called `n_iter_vs_n_world_nodes_3_layers_500_trials_iso_count.pdf` which corresponds to figure 7 in the paper. Other figures related to time and number of isomorphisms will also be generated.

### To run the full Ensemble-Sequential Experiments

Running the experiments will take a while depending on your hardware.

```bash
$ cd experiments
$ python run_erdos_renyi.py
$ python plot_erdos_renyi.py
```
Change the variables in run_erdos_renyi.py to run with different settings i.e. number of layers and whether isomorphism counting is being done.

plot_erdos_renyi.py will generate a figure called `n_iter_vs_n_world_nodes_3_layers_500_trials_iso_count.pdf` which corresponds to figure 7 in the paper. Other figures related to time and number of isomorphisms will also be generated.

### To run the benchmarking methods mentioned in the paper

Running the experiments will take a while depending on your hardware.

```bash
$ cd experiments
$ python run_erdos_renyi.py
$ python plot_erdos_renyi.py
```
Change the variables in run_erdos_renyi.py to run with different settings i.e. number of layers and whether isomorphism counting is being done.

plot_erdos_renyi.py will generate a figure called `n_iter_vs_n_world_nodes_3_layers_500_trials_iso_count.pdf` which corresponds to figure 7 in the paper. Other figures related to time and number of isomorphisms will also be generated.


### Synthetic Datasets

The example runs could be found in example.py, which runs through one of the 90 synthetic network datasets we created.
To run through the synthetic networks, please download them through the Google Drive Link here: 

https://drive.google.com/drive/folders/1sfycenFPrYXBHSUlJ7ovEIYHFY5-mGg2?usp=drive_link


### Real World Datasets

The real world networks could be found under the following links, due to copy right reasons, we will only show the link to download them:
The following is taken from ICON: https://icon.colorado.edu/#!/networks

- chess: Search for Kaggle chess players (2010) on : https://icon.colorado.edu/#!/networks
- bitcoin: Bitcoin Alpha trust network (2017): https://snap.stanford.edu/data/soc-sign-bitcoinalpha.html
- obitcoin: Bitcoin OTC trust network (2017): https://snap.stanford.edu/data/soc-sign-bitcoinotc.html
- obrazil: Brazilian prostitution network (2010): http://konect.cc/networks/escorts/
- london: London bike sharing (2014): https://github.com/konstantinklemmer/bikecommclust
- mit: Search for Reality mining proximity network (2004) on: https://icon.colorado.edu/#!/networks
- radoslaw:  Search for Manufacturing company email (2010) on: https://icon.colorado.edu/#!/networks

The following is taken from network repository:
- ant1-ant6: https://networkrepository.com/asn.php (see insect-ant-colony)
- emaildnc: https://networkrepository.com/email-dnc.php
- fbforum: https://networkrepository.com/fb-forum.php
- fbmsg: https://networkrepository.com/fb-messages.php

The following is given to us by the authors, special thanks to the authors for sharing the data. 
- bionet1-2: https://www3.nd.edu/~tmilenko/software_data.html
- Khalique Newaz and Tijana Milenkovic (2020), Improving inference of the dynamic biological network underlying aging via network propagation, IEEE/ACM Transactions on Computational Biology and Bioinformatics, DOI: 10.1109/TCBB.2020.3022767.


### Previous Mistakes

You could also find the past bugged version of the code both in the same folder and on Github for debugging purposes. The noticable change could be found in the Github history.
