This repo contains the official implementation of our paper: "LASIL: Learner-Aware Supervised Imitation Learning For Long-term Microscopic Traffic Simulation". 
  
**CVPR 2024**  


# Installation 

### Environment
```shell
pip install -r requirement.txt
```


### Data

Download the csv file from [Pneuma Dataset](https://open-traffic.epfl.ch/index.php/downloads/); only the files in ```All Drones``` are needed. 
Store all files according to its days in data/1024, data/1029, data/1030,data/1101.
      
### Training 


Run ```train.py``` to learn the model which will do the data preprocessing firstly (taking a day). You need to specify the model name ```--model_name```.
```shell
python train.py --model_name lasil
```

### Evaluation   

Run ```eval.py``` to do closed-loop testing. You need to specify the pretrained model path ```--ckpt_path```. 
```shell
python eval.py --model_name lasil --ckpt_path path_to_pretrained_model
```

## Citation

```
@inproceedings{guo2024end,
  title={LASIL: Learner-Aware Supervised Imitation Learning For Long-term Microscopic Traffic Simulation},
  author={Ke Guo, Zhenwei Miao, Wei Jing, Weiwei Liu, Weizi Li, Dayang Hao, Jia Pan },
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
