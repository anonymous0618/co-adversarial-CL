# co-adversarial-CL

To run co-adversarial CL model with BiasMF and LightGCN, use the following command
```
python Main.py --reg 1e-7 --advBat 16 --save_path modelName
```

Some important hyperparameters:
* reg: weight for the weight-decay regularization. Recommended values: 1e-8, 1e-7, 1e-6
* advBat: number of sampled items from which to find out i_G


BiasMF and LightGCN are implemented in Main_singleMethod.py, change the called object (self.LightGCN, self.BiasMF) in trainEpoch, testEpoch to change the model.
