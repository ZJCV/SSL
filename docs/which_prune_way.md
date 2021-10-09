
# Which Prune Way

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

Take `filter_wise` as an example. The regularization coefficient is `1e-5`. 

The results of training on `CIFAR100` are as follows



## Pruning

Test five different prune way:

```
def computer_weight(weight, prune_way, dimension):
    if prune_way == 'group_lasso':
        return group_lasso_by_filter_or_channel(weight.data, dimension)
    elif prune_way == 'mean_abs':
        return torch.mean(weight.data.abs(), dim=dimension)
    elif prune_way == 'mean':
        return torch.mean(weight.data, dim=dimension)
    elif prune_way == 'sum_abs':
        return torch.sum(weight.data.abs(), dim=dimension)
    elif prune_way == 'sum':
        return torch.sum(weight.data, dim=dimension)
    else:
        raise ValueError(f'{prune_way} does not exists')
```

The results of pruning are as follows

|     arch    |  prune way  | pruning ratio | actual pruning ratio | flops/G | model size/MB | Flops after pruning | Model size after pruning |
|:-----------:|:-----------:|:-------------:|:--------------------:|:-------:|:-------------:|:-------------------:|:------------------------:|
| vggnet16_bn | group_lasso |      20%      |        18.56%        |  15.51  |     134.68    |         7.67        |          130.88         |
| vggnet16_bn |   mean_abs  |      20%      |        19.13%        |  15.51  |     134.68    |         9.20        |          129.51          |
| vggnet16_bn |     mean    |      20%      |        19.13%        |  15.51  |     134.68    |        10.92        |           69.38          |
| vggnet16_bn |   sum_abs   |      20%      |        19.13%        |  15.51  |     134.68    |         6.75        |          132.22          |
| vggnet16_bn |     sum     |      20%      |        19.32%        |  15.51  |     134.68    |        13.85        |           55.88          |

## Fine-tuning

The results of fine-tuning on `CIFAR100` are as follows

|     arch    |  prune way  |  top1  |  top5  |
|:-----------:|:-----------:|:------:|:------:|
| vggnet16_bn | group_lasso | 80.810 | 95.090 |
| vggnet16_bn |   mean_abs  | 80.470 | 94.940 |
| vggnet16_bn |     mean    | 79.650 | 94.800 |
| vggnet16_bn |   sum_abs   | 79.440 | 94.880 |
| vggnet16_bn |     sum     | 79.580 | 95.100 |

## Summarize

From above statistics, `group_lasso` get the better result.