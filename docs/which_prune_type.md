
# Which Prune Type

## Base

Using `VGGNet16_BN`, the original parameters are as follows:

```angular2html
FLOPs= 15.507270656G 
params= 134.678692M 
```

The results of training on `CIFAR100` are as follows

```angular2html
top1 acc: 80.940  top5 acc: 95.550
```

## Training

Test three different prune way (`filter_wise/channel_wise/filter_and_channel_wise`). Training with filter pruning, the regularization coefficient is `1e-5`

The results of training on `CIFAR100` are as follows

|     arch    |       prune  type       | regularization ratio |  top1  |  top5  |
|:-----------:|:-----------------------:|:--------------------:|:------:|:------:|
| vggnet16_bn |       filter_wise       |         1e-5         | 80.790 | 95.310 |
| vggnet16_bn |       channel_wise      |         1e-5         | 80.790 | 95.310 |
| vggnet16_bn | filter_and_channel_wise |         1e-5         | 80.770 | 95.580 |

## Pruning

Take `group_lasso/filter_wise` as prune way . The results of pruning are as follows

|     arch    |        prune type       |  prune way  | pruning ratio | actual pruning ratio | flops/G | model size/MB | Flops after pruning | Model size after pruning |
|:-----------:|:-----------------------:|:-----------:|:-------------:|:--------------------:|:-------:|:-------------:|:-------------------:|:------------------------:|
| vggnet16_bn |       filter_wise       | group_lasso |      20%      |        18.56%        |  15.51  |     134.68    |         7.67        |          130.88          |
| vggnet16_bn |       filter_wise       |   mean_abs  |      20%      |        19.13%        |  15.51  |     134.68    |         9.20        |          129.51          |
| vggnet16_bn |       channel_wise      | group_lasso |      20%      |        18.95%        |  15.51  |     134.68    |         8.21        |          131.26          |
| vggnet16_bn |       channel_wise      |   mean_abs  |      20%      |        19.38%        |  15.51  |     134.68    |         9.57        |          130.53          |
| vggnet16_bn | filter_and_channel_wise | group_lasso |      20%      |        13.91%        |  15.51  |     134.68    |         9.56        |          131.94          |
| vggnet16_bn | filter_and_channel_wise |   mean_abs  |      20%      |        11.29%        |  15.51  |     134.68    |        11.35        |          131.84          |
## Fine-tuning

The results of fine-tuning on `CIFAR100` are as follows

|     arch    |        prune type       |  prune way  |  top1  |  top5  |
|:-----------:|:-----------------------:|:-----------:|:------:|:------:|
| vggnet16_bn |       filter_wise       | group_lasso | 80.810 | 95.090 |
| vggnet16_bn |       filter_wise       |   mean_abs  | 80.470 | 94.940 |
| vggnet16_bn |       channel_wise      | group_lasso | 80.800 | 95.070 |
| vggnet16_bn |       channel_wise      |   mean_abs  | 80.660 | 95.280 |
| vggnet16_bn | filter_and_channel_wise | group_lasso | 80.530 | 95.320 |
| vggnet16_bn | filter_and_channel_wise |   mean_abs  | 80.720 | 95.130 |

## Summarize

From above statistics, `Filter_wise` prune way made the better result.