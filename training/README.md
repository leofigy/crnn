### Usage 

-> Training
```
python .\train.py fotos
```

-> Inference (testing the model)
```
 python .\inference.py .\initial_model\baches_cfg20200103T2138\mask_rcnn_baches_cfg_0005.h5 .\fotos\images\90.jpg
```

-> splitting the images
```
 python .\spliter.py .\staging\nestorvideo.mp4
```

-> Stream 
```
python .\stream.py .\initial_model\baches_cfg20200103T2138\mask_rcnn_baches_cfg_0005.h5 .\staging\
```

Author @nestorNeo