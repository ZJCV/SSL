<div align="right">
  è¯­è¨:
    ð¨ð³
  <a title="è±è¯­" href="./README.md">ðºð¸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/SSL"><img align="center" src="./imgs/SSL.png"></a></div>

<p align="center">
  Â«SSLÂ»å¤ç°äºè®ºæ<a title="" href="https://arxiv.org/abs/1608.03665">Learning Structured Sparsity in Deep Neural Networks</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

* è§£æï¼[ Learning Structured Sparsity in Deep Neural Networks](https://blog.zhujian.life/posts/67852044.html)

é¤äºè®ºææåçå ç§åªææ¹å¼ï¼`æ»¤æ³¢å¨åªæ/ééåªæ/æ»¤æ³¢å¨_ééåªæ/å±åªæ`ï¼å¤ï¼æ¬ä»åºè¿æµè¯äºä¸åæéå½æ°ï¼`group_lasso/mean_abs/mean/sum_abs/sum`ï¼å¯¹äºåªæçå½±åã

* [Which Prune Type?](./docs/which_prune_type.md)
* [Which Prune Way?](./docs/which_prune_way.md)

æ´è¯¦ç»çè®­ç»æ°æ®å¯ä»¥æ¥çï¼

* [Details](./docs/details.md)

## åå®¹åè¡¨

- [åå®¹åè¡¨](#åå®¹åè¡¨)
- [èæ¯](#èæ¯)
- [å®è£](#å®è£)
- [ç¨æ³](#ç¨æ³)
- [ä¸»è¦ç»´æ¤äººå](#ä¸»è¦ç»´æ¤äººå)
- [è´è°¢](#è´è°¢)
- [åä¸è´¡ç®æ¹å¼](#åä¸è´¡ç®æ¹å¼)
- [è®¸å¯è¯](#è®¸å¯è¯)

## èæ¯

åºäº`Group Lasso`ï¼`SSL`å®ç°äºæ»¤æ³¢å¨/éé/æ»¤æ³¢å¨å½¢ç¶/å±åªæåè½ã

## å®è£

```
$ pip install -r requirements.txt
```

## ç¨æ³

é¦åï¼è®¾ç½®ç¯å¢åé

```angular2html
$ export PYTHONPATH=<project root path>
$ export CUDA_VISIBLE_DEVICES=0
```

ç¶åè¿è¡`è®­ç»-åªæ-å¾®è°`

* è®­ç»

```
$ python tools/train.py -cfg=configs/vggnet/vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_filter_wise_1e_5.yaml
```

* åªæ

```angular2html
$ python tools/prune/prune_vggnet.py
```

* å¾®è°

```angular2html
$ python tools/train.py -cfg=configs/vggnet/refine_mean_abs_0_2_vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_filter_wise_1e_5.yaml
```

æåï¼å¨éç½®æä»¶ç`PRELOADED`éé¡¹ä¸­è®¾ç½®å¾®è°åçæ¨¡åè·¯å¾

```angular2html
$ python tools/test.py -cfg=configs/vggnet/refine_mean_abs_0_2_vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_filter_wise_1e_5.yaml
```

## ä¸»è¦ç»´æ¤äººå

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## è´è°¢

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
## åä¸è´¡ç®æ¹å¼

æ¬¢è¿ä»»ä½äººçåä¸ï¼æå¼[issue](https://github.com/ZJCV/SSL/issues)ææäº¤åå¹¶è¯·æ±ã

æ³¨æ:

* `GIT`æäº¤ï¼è¯·éµå®[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)è§è
* è¯­ä¹çæ¬åï¼è¯·éµå®[Semantic Versioning 2.0.0](https://semver.org)è§è
* `README`ç¼åï¼è¯·éµå®[standard-readme](https://github.com/RichardLitt/standard-readme)è§è

## è®¸å¯è¯

[Apache License 2.0](LICENSE) Â© 2021 zjykzj