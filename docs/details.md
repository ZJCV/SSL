
# Training Results

## VGGNet

`cifar100 + e100 + sgd + mslr + ssl(1e-5)`

```
# vggnet16_bn default training
total -  top1 acc: 80.940  top5 acc: 95.550
# vggnet16_bn channel_wise pruning training
total -  top1 acc: 80.790  top5 acc: 95.310
# vggnet16_bn filter_and_channel_wise pruning training
total -  top1 acc: 80.770  top5 acc: 95.580
# vggnet16_bn filter_and_channel_wise pruning training
total -  top1 acc: 80.790  top5 acc: 95.310
```

## ResNet

`cifar100 + e100 + sgd + mslr + ssl(1e-5)`

```
# resnet50 default training
total -  top1 acc: 84.070  top5 acc: 96.290
# resnet50 depth_wise pruning training
total -  top1 acc: 84.070  top5 acc: 96.290
```

## Filter_wise

|     arch    |  prune type |  prune way  | pruning ratio | actual pruning ratio | flops/G | model size/MB | Flops after pruning | Model size after pruning |  top1  |  top5  |
|:-----------:|:-----------:|:-----------:|:-------------:|:--------------------:|:-------:|:-------------:|:-------------------:|:------------------------:|:------:|:------:|
| vggnet16_bn | filter_wise | group_lasso |      20%      |        18.56%        |  15.51  |     134.68    |         7.67        |          130.88          | 80.810 | 95.090 |
| vggnet16_bn | filter_wise |   mean_abs  |      20%      |        19.13%        |  15.51  |     134.68    |         9.20        |          129.51          | 80.470 | 94.940 |
| vggnet16_bn | filter_wise |     mean    |      20%      |        19.13%        |  15.51  |     134.68    |        10.92        |           69.38          | 79.650 | 94.800 |
| vggnet16_bn | filter_wise |   sum_abs   |      20%      |        19.13%        |  15.51  |     134.68    |         6.75        |          132.22          | 79.440 | 94.880 |
| vggnet16_bn | filter_wise |     sum     |      20%      |        19.32%        |  15.51  |     134.68    |        13.85        |           55.88          | 79.580 | 95.100 |
| vggnet16_bn | filter_wise |   mean_abs  |      40%      |        39.01%        |  15.51  |     134.68    |         6.42        |          112.47          | 78.900 | 94.470 |
| vggnet16_bn | filter_wise |   mean_abs  |      40%      |        58.71%        |  15.51  |     134.68    |         4.60        |           74.06          | 75.880 | 93.090 |

## Channel_wise

|     arch    |  prune type  |  prune way  | pruning ratio | actual pruning ratio | flops/G | model size/MB | Flops after pruning | Model size after pruning |  top1  |  top5  |
|:-----------:|:------------:|:-----------:|:-------------:|:--------------------:|:-------:|:-------------:|:-------------------:|:------------------------:|:------:|:------:|
| vggnet16_bn | channel_wise | group_lasso |      20%      |        18.95%        |  15.51  |     134.68    |         8.21        |          131.26          | 80.800 | 95.070 |
| vggnet16_bn | channel_wise |   mean_abs  |      20%      |        19.38%        |  15.51  |     134.68    |         9.57        |          130.53          | 80.660 | 95.280 |
| vggnet16_bn | channel_wise |   mean_abs  |      40%      |        38.98%        |  15.51  |     134.68    |         6.15        |          126.91          | 79.900 | 94.910 |
| vggnet16_bn | channel_wise |   mean_abs  |      60%      |        59.00%        |  15.51  |     134.68    |         4.11        |          123.30          | 78.620 | 94.480 |

## Filter_and_Channel_wise

|     arch    |        prune type       |  prune way  | pruning ratio | actual pruning ratio | flops/G | model size/MB | Flops after pruning | Model size after pruning |  top1  |  top5  |
|:-----------:|:-----------------------:|:-----------:|:-------------:|:--------------------:|:-------:|:-------------:|:-------------------:|:------------------------:|:------:|:------:|
| vggnet16_bn | filter_and_channel_wise | group_lasso |      20%      |        13.91%        |  15.51  |     134.68    |         9.56        |          131.94          |  80.53 | 95.320 |
| vggnet16_bn | filter_and_channel_wise |   mean_abs  |      20%      |        11.29%        |  15.51  |     134.68    |        11.36        |          131.84          | 80.720 | 95.130 |
| vggnet16_bn | filter_and_channel_wise |   mean_abs  |      40%      |        22.98%        |  15.51  |     134.68    |         8.98        |          128.81          | 80.040 | 95.180 |
| vggnet16_bn | filter_and_channel_wise |   mean_abs  |      60%      |        35.07%        |  15.51  |     134.68    |         7.45        |          125.67          | 79.570 | 94.920 |

## Depth_wise

|   arch   | prune type |  prune way  | N | flops/G | model size/MB | Flops after pruning | Model size after pruning |  top1  |  top5  |
|:--------:|:----------:|:-----------:|:-:|:-------:|:-------------:|:-------------------:|:------------------------:|:------:|:------:|
| resnet50 | depth_wise | group_lasso | 1 |   4.11  |     23.72     |         3.89        |           23.64          | 83.610 | 95.970 |
| resnet50 | depth_wise |   mean_abs  | 1 |   4.11  |     23.72     |         3.89        |           19.25          | 83.630 | 96.090 |
| resnet50 | depth_wise |   mean_abs  | 2 |   4.11  |     23.72     |         3.67        |           14.79          | 82.990 | 95.710 |
| resnet50 | depth_wise |   mean_abs  | 4 |   4.11  |     23.72     |         3.23        |           13.39          | 82.280 | 95.400 |