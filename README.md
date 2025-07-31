# SurfFold: A Unified Model for Protein Inverse Folding by Integrating Surface and Structural Information
## Environment Preparation  
```
git clone git@github.com:jiudizhengf/SurfFold.git  
cd SurfFold  
conda env create -f environment.yaml
```
## Dataset
 we utilize the CATH4.2 dataset. CATH4.2 is a large-scale database for protein structure classification and analysis, providing hierarchical annotations for tens of thousands of protein structures. It is widely used in structural bioinformatics, protein folding prediction, and functional studies. We downloaded the 40\% non-redundant subset of CATH4.2 from the official website. In addition, we strictly followed the data partitioning protocol used by previous inverse folding methods. Specifically, the dataset was divided into a training set, validation set, and test set, containing 18,024, 608, and 1,120 samples, respectively. Furthermore, to facilitate more extensive evaluation, we created additional test subsets based on the original test set. These include a short-chain test set with sequences shorter than 100 residues and a single-chain test set with proteins containing only a single chain.  
## Data Process
The official original dataset can be downloaded from [http://pdbbind.org.cn/](https://www.cathdb.info/). Then create a directory named `CATH4.2` and place the raw data according to the `spilit.json` partitions within it.If you prefer to skip data processing, preprocessed data is available for direct download.  
## train and test the model
You can initiate training using the base configuration in `inverse_folding.py`, where the hyperparameters serve as the initial training settings. After data preparation, simply execute `run  inverse_folding.py` to begin training and testing.