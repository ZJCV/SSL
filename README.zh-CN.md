<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/SSL"><img align="center" src="./imgs/SSL.png"></a></div>

<p align="center">
  «SSL»复现了论文<a title="" href="https://arxiv.org/abs/1608.03665">Learning Structured Sparsity in Deep Neural Networks</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

基于`Group Lasso`，`SSL`实现了滤波器/通道/滤波器形状/层剪枝功能。

## 安装

```
$ pip install -r requirements.txt
```

## 用法

首先，设置环境变量

```angular2html
$ export PYTHONPATH=<project root path>
$ export CUDA_VISIBLE_DEVICES=0
```

然后进行`训练-剪枝-微调`

* 训练

```
$ python tools/train.py -cfg=configs/vggnet/vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_filter_wise_1e_5.yaml
```

* 剪枝

```angular2html
$ python tools/prune/prune_vggnet.py
```

* 微调

```angular2html
$ python tools/train.py -cfg=configs/vggnet/refine_mean_abs_0_2_vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_filter_wise_1e_5.yaml
```

最后，在配置文件的PRELOADED选项中设置微调后的模型路径

```angular2html
$ python tools/test.py -cfg=configs/vggnet/refine_mean_abs_0_2_vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_filter_wise_1e_5.yaml
```

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

* [lionminhu/structured-sparsity-learning](https://github.com/lionminhu/structured-sparsity-learning)

```
@misc{wen2016learning,
      title={Learning Structured Sparsity in Deep Neural Networks}, 
      author={Wei Wen and Chunpeng Wu and Yandan Wang and Yiran Chen and Hai Li},
      year={2016},
      eprint={1608.03665},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```
## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/ZJCV/SSL/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2021 zjykzj