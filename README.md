# DualFed: Enjoying both Generalization and Personalization in Federated Learning via Hierachical Representations
This is the PyTorch implemention of our paper DualFed: Enjoying both Generalization and Personalization in Federated Learning via Hierachical Representations, in ACM Multimedia 2024.

Paper link：https://arxiv.org/abs/2407.17754
## Abstract

In personalized federated learning (PFL), it is widely recognized that achieving both high model generalization and effective personalization poses a significant challenge due to their conflicting nature. As a result, existing PFL methods can only manage a trade-off between these two objectives. This raises an interesting question: Is it feasible to develop a model capable of achieving both objectives simultaneously? Our paper presents an affirmative answer, and the key lies in the observation that deep models inherently exhibit hierarchical architectures, which produce representations with various levels of generalization and personalization at different stages. A straightforward approach stemming from this observation is to select multiple representations from these layers and combine them to concurrently achieve generalization and personalization. However, the number of candidate representations is commonly huge, which makes this method infeasible due to high computational this http URL address this problem, we propose DualFed, a new method that can directly yield dual representations correspond to generalization and personalization respectively, thereby simplifying the optimization task. Specifically, DualFed inserts a personalized projection network between the encoder and classifier. The pre-projection representations are able to capture generalized information shareable across clients, and the post-projection representations are effective to capture task-specific information on local clients. This design minimizes the mutual interference between generalization and personalization, thereby achieving a win-win situation. Extensive experiments show that DualFed can outperform other FL methods.
![image](https://github.com/GuogangZhu/DualFed/blob/master/fig/Framework.png)

## Download

```
git clone https://github.com/GuogangZhu/DualFed.git DualFed
```

## Setup

See the `requirements.txt` for environment configuration.

```python
pip install -r requirements.txt
```

## Dataset
Please download the datasets from their official website

## Examples
We provide some running examples in ./run/. You can directly run these script by specify the data direction.
### PACS
  ```
  cd ./run
  sh DualFed_PACS.sh 'your data direction'
  ```
### DomainNet
  ```
  cd ./run
  sh DualFed_DomainNet.sh 'your data direction'
  ```
### OfficeHome
  ```
  cd ./run
  sh DualFed_OfficeHome.sh 'your data direction'
  ```

### Description of Arguments

- **con_temp:** temperature coefficient $\tau$ in contrastive loss

- **con_lambda:** the coefficient $\lambda$ used to balance the contrastive and cross-entropy loss

## Citation

If you make advantage of DualFed in your research, please cite the following in your manuscript:

```latex
@inproceedings{DualFed,
  title={DualFed: Enjoying both Generalization and Personalization in Federated Learning via Hierachical Representations},
  author={Guogang Zhu, Xuefeng Liu, Jianwei Niu, Shaojie Tang, Xinghao Wu, Jiayuan Zhang},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024}
}
```
