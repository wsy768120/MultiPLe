## Environment

Linux's system with CUDA devices required.

### Python packages

```
conda create --name multiple python=3.8
conda activate multiple
conda install -c pytorch pytorch=1.10.0 cudatoolkit=11.3
conda install -c huggingface transformers=4.11.3 
conda install -c conda-forge spacy=3.2.0 cupy=9.6.0
conda install numpy scikit-learn tqdm pandas
python -m spacy download en_core_web_sm
```

### Download models and tools
- put the models under path ```pretrain/plm/```
```
pretrain
├─plm
│  └─xlm-roberta-large
│  └─Helsinki-NLP
│    └─opus-mt-en-es
│    └─opus-mt-en-zh
```
- put the tools under path ```tool/```
```
tool
├─word2id.json
├─word_embedding.npy
```

Or you could edit code in ```tool/pretrain_model_helper.py``` to download automatically when running code.

## Experiments

### pre-process data
- Retrive full version of data from dataset soure and put them under path ```data/Ace``` and ```data/FewShotED```.
```
data
├─Ace
│      dev_docs.json
│      test_docs.json
│      train_docs.json
│
└─FewShotED
        Few-Shot_ED.json
```
- Run follow command to process data and save it to path ```out/processed_data```
```
python -m data_process.gen_train_data
```


### Train model

```
python train.py
```

Edit the code in ```train.py```  to change the setting of Experiments.

The trained model would be saved to path ```out/checkpoints```

### Test model

```
python test.py
```


You need first add model checkpoints name to main function of ```test.py``` .


