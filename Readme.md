### Sequential Stacking Link Prediction Algorithms for Temporal Networks 

This is the github repo accompany the [paper by Xie He, Amir Ghasemian, Eun Lee, Aaron Clauset, Peter Mucha](https://www.researchsquare.com/article/rs-2535525/v1). 
The paper is currently under revision at Nature Communications. 

**Please cite the paper when using the data or code. See License Information for more details on Usage.**

### System requirements

Use the commend

```bash
conda env create -f environment.yml
```

to create the required environmnet for the code (Only if you want to use the full Ensemble-Sequential, because of the dependency of E-LSTM-D, Time Series, and T-SBM).
Note very importantly, currently because of the GNU and GCC packages that are dependencies required from E-LSTM-D, it is only possible to do the full Ensemble-Sequential on Linux system.

If you wish to run only the Top-Sequential method with the topological features, you could instead do:

```bash
pip install scipy numpy pandas networkx scikit-learn imblearn
```

If you further with to run only Top-Sequential and Time Series, then you should also install:

```bash
pip install statsmodels
```


This has been tested on all the popular platforms and should work for Windows, Mac OS, Linux, Google Colab, etc.

To reproduce all results from our experiments, you will need at least Python 3.10 and a few packages installed(see the enviornment file for specific details). 

You can check your python version with

```bash
$ python --version
```

### To run only the Top-Sequential Experiments

The best way to run only the Top-Sequential Experiment is to follow the `example.py` file. 

```bash
$ python example.py
```
Change the variables and/or numbers in `example.py` to change the corresponding variables in the paper. 

Note that you have to manually determine the number of layers you want the algorithm to work with. 

- The search variable **u** could be found and replaced in `edges_orig = edges_orig[0:u]` (6 in all of our experiment)
- The flow variable **q** could be found and replaced in `predict_num = q` (3 in all of our experiment)

Running `example.py` (which contain two functions) will generate two AUC scores, accordingly with the partially observed case and the completely unobserved case in the paper. 

### To run the full Ensemble-Sequential Experiments

Running the experiments will take a while depending on your hardware. In particular, both E-LSTM-D and T-SBM could be a bit slow even for smaller networks. 

We describe the process for the partially observed case, the completely unobserved case is done in the exact same setting, but with slightly different names. 
To run the full Ensemble-Sequential experiment. You have to first:

1. Download and install the code and relevant packages from: [E-LSTM-D](https://github.com/jianz94/e-lstm-d)
2. Download and install the code and relevant packages from: [T-SBM](https://github.com/seeslab/MMmultilayer)
3. Make sure you have installed the required environment using the `environment.yml`

First, you have to run the E-LSTM-D codes in order to get the features and AUC scores from it. 

```bash
$ cd ensemble_with_others/E-LSTM-D/Partially-observed
$ python convert_partial.py
$ python generate_output.py
$ python calculate_elstmd.py
```
this will in turn gives you a full feature matrix from E-LSTM-D, which you could used to stack with the topological features extracted with Top-Sequential method. 

Then, you have to run the Time-Series codes in order to get the features from them. 

```bash
$ cd ensemble_with_others/Time-Series
$ python run_time_series.py
```

After that, navigate towards the folder `ensemble_with_others/Ensemble_final_edition/`.

Once inside the folder you have to first generate the feature matrix for the dataset first. You can do this by:

```bash
$ python data_runner.py # this will create the T-SBM features (which would be an edge indicator) and the Toplogical features
$ python create_lstm_df.py # this will create the LSTM features
$ python process_ts.py # this will create the time series features and add them to the end of the previous features. 
```

If done correctly, you should be seeing folders named "lstm_feature", "for_sbm", "feature_metrices", "results", "all_features", "edge_tf_true", "edge_tf_tr", "ef_gen_tr", "ef_gen_tr". 

Note: you might encounter folder not found error, in which case you should check the folder name in the code and make sure to change that manually. 

TODO: fix this so that it could be done automatically. 

Then you could go ahead and call:

```
$ python calculate_different_AUC.py 
```

This will give you the complete AUC scores result of the dataset you desired. Please change the `feat_path` in the code to your desired output path.

Very importantly, the AUC scores order that you will end up getting after the partially observed case should be: 

```
auc_methods = ['Top-Sequential-Stacking', 'Time-Series', 'Tensorial-SBM', 'E-LSTM-D', 'Ensemble-Sequential-Stacking',]
```

The AUC scores order that you will get after the completely unobserved case will be the same order, except that you will ignore the third column, `Tensorial-SBM`, because that would be a meaningless result that is repeating the partially observed case.  


Note also: feel free to use this ensemble learning method stacked with other features of your liking. Theoritically any features that could generated with a partially observed network would work with that case, and note also completely unobserved case would require features that could be generated from the previous time slot. 


Note also that if you do not wish the run the full E-LSTM-D and Time Series, but are only interested in the toplogical feature + T-SBM, you could simply navigate to the folder, and run:

```
$ python data_runner.py 
```

This will give you all the feature matrix you need to further use your preferred algorithm. 

If done correctly, you should see: "for_sbm", "feature_metrices", "results", "edge_tf_true", "edge_tf_tr", "ef_gen_tr", "ef_gen_tr". 

And you could then call 

```
$ python calculate_different_AUC.py 
```

You could get both the Top-Sequential AUC and the T-SBM AUC without the trouble of installing E-LSTM-D. 

Namely, the choice `0` gives you Top-Sequential AUC, choice `1` gives you Time-Series, choice `2` gives you T-SBM, choices `3` gives you E-LSTM-D, and choice `4` gives you Ensemble-Sequential-Stacking, just like what is described above. 

But you need to change the variable `feat_path` to your own feature path before proceed if you have **not** run the other two. If you have run the other two in the order described above, you are good to go. 

And in the case you have **not** run neither Time-Series nor E-LSTM-D, you have to only the choice of `0` and `2`. Any other option will likely give you an error message. This part has been tested on Linux and Windows and Mac. 

If there's any question, feel free to leave a message on Github or email directly. 


### To run the benchmarking methods mentioned in the paper individually

For E-LSTM-D:

1. Download and install the code and relevant packages from: [E-LSTM-D](https://github.com/jianz94/e-lstm-d)
2. Either you could then run their code directly to caluclate the AUC.
3. Or you could directly run the full Ensemble-Sequential code, which automatically generate the AUC scores after the full-run.

For Tensorial-SBM:

1. Download and install the code and relevant packages from: [T-SBM](https://github.com/seeslab/MMmultilayer)
2. Either you could then run their code directly to caluclate the AUC.
3. Or you could directly run the full Ensemble-Sequential code, which automatically generate the AUC scores after the full-run.

For Time Series: 

1. Download and install the code and relevant packages and navigate to the folder: ensemble_with_others/Time-Series. 
3. Modify and run the full Ensemble-Sequential code, which automatically generate the AUC scores after the full-run.

### Synthetic Datasets

The example runs could be found in example.py, which runs through one of the 90 synthetic network datasets we created.
To run through the synthetic networks, please download them through the [Google Drive Link](https://drive.google.com/drive/folders/1sfycenFPrYXBHSUlJ7ovEIYHFY5-mGg2?usp=drive_link) here. 
Once downloaded, go ahead and extract the folder into the same folder under `TOLP.py` and change the path name in the `example.py` and/or modify to your liking. 

Note that the naming of the synthetic networks could be very confusing. Here we list the naming pattern for both types of synthetic network so that the readers are not confused. We did the naming this way to avoid long and arduous names of the files.
For the naming convention, see the functions in the python file `translate.py` for specific details. 


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
