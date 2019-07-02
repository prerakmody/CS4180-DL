# 1. Topic - Efficient Deep Learning - Model Compression via pruning forobject detection
 - __Problem Statement__
    - Deep Neural Networks are both memory and compute intensive and hence are not a feasible option for embedded devices. Various methods for “Efficient Deep Learning” such as model pruning, model estimation, and student-teacher methods have been proposed to abate this problem for both model compression and acceleration in image classification tasks
    - Do model compression techniques such as weight and filter pruning produce similar performance gains on object detection models as they have for image classification models? In our viewpoint, similar performance gains would imply that the compressed models perform at least as good as their uncompressed counterparts. 
 - __Test Frameworks__
    - Model : [YOLOv2](https://arxiv.org/abs/1612.08242) ([Code](https://github.com/marvis/pytorch-yolo2))
    - Dataset : [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) 
 - __Methods__ : 
    - [Weight pruning : Learning both Weights and Connections for Efficient Neural Networks, NIPS, 2015 (Han et.al)](https://dl.acm.org/citation.cfm?id=2969366)
    - [Filter Pruning : Pruning Filters for Efficient ConvNets, Arxiv, 2016 (Li et.al) ](https://arxiv.org/abs/1608.08710)
 - __Steps__
    1. Perform either of the above pruning techniques (weight or filter) on a pretrained network and evaluate mAP (mAP-Pruned)
    2. Retrain the pruned network and evalute (mAP-Retrained) 
 - __Conclusion__
    - With retraining, even aggresively pruned networks have only a small loss in mAP
    - Although, weight pruning performs better than filter pruning, physically removing pruned filters is a more feasible task than removing specific pruned weights 

# 2. Run
 - Refer to [demo_yolov2_retrain_colab.ipynb](demo/demo_retrain/demo_yolov2_retrain_colab.ipynb)
 - You can run both train and predict operations there on any pruned weights
 - To make the code files easier to read, comments with enumerated steps are provided 

# 3. Insights
 - Results
    - ![Filter Pruning](https://github.com/prerakmody/CS4180-DL/blob/feature/pre-master/demo/demo_retrain/results/ModelCompression_PASCAL2007_YOLOv2_FilterPruning.png)
    - ![Weight Pruning](https://github.com/prerakmody/CS4180-DL/blob/feature/pre-master/demo/demo_retrain/results/ModelCompression_PASCAL2007_YOLOv2_WeightPruning.png)