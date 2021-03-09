# 5*E
This is the code for implementation of [5* Knowledge Graph Embeddings with Projective Transformations](https://arxiv.org/abs/2006.04986) (AAAI 2021).

## Installation of baseline

The score functions of the following models are inlcuded in this code:

**Models**
- [x] CP
- [x] ComplEx
- [x] 5*E


The starting point is to install kbc framework. For which you need to create a conda environment with pytorch cython and scikit-learn :
```
conda create --name kbc_env python=3.7
source activate kbc_env
conda install --file requirements.txt -c pytorch
```

Then, in the next step, install the kbc package to this environment
```
python setup.py install
```

## Datasets

To download the datasets, go to the kbc/scripts folder and run:
```
chmod +x download_data.sh
./download_data.sh
```

Once the datasets are download, add them to the package data folder by running :
```
python kbc/process_datasets.py
```

This will create the files required to compute the filtered metrics.

## Running the code
Reproduce the results below with the following command :
```
python kbc/learn.py --dataset datasetName --model FiveStarE --rank dimension --optimizer
Adagrad --learning_rate lr --batch_size batchSize --regularizer N3 --reg regularizerValue
 --max_epochs EpochNumber --valid 50
```
## 5*E: Covering simultaneous transformations: translation, rotation, homothety, reflection and inversion.

Existing state of the art KGE models namely TransE, RotatE, ComplEx and QuatE cover only one or two transformation types among translation, rotation and homothety.


<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/OtherTransfType.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>


<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/default.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>

<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/flows.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>


<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/entities.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>



<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/relation1.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>



<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/relation2.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>

<iframe width="560" height="315" src="https://www.youtube.com/embed/TP0MKNi7Q7w" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## License
kbc is CC-BY-NC licensed, as found in the LICENSE file
