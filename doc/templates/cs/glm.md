# LLM系列之GLM

## Abstract

截至2024年7月，智谱AI陆续发布的GLM系列大模型有：

|发布日期|模型版本|主要亮点|
|--|--|--|
|Jan 16, 2024|文本模型：GLM4-9b, GLM4-9b-chat, GLM4-9B-Chat-1M<br>多模态模型：GLM4V-9b||
|August 4, 2022|基座模型GLM-130B||

模型效果上，GLM4-9b整体优于LLama3-8B和Llama3-8B-Instruct，完整效果可查看大模型榜单。

本文主要讨论：

- 基座模型GLM
    - GLM所谓的“通用”在预训练任务上是如何体现出来的？
    - GLM的模型架构在Transformer基础上有哪些改动？
- 对话模型
    - GLM4有哪些主要的贡献和核心的亮点？
    - GLM4的效果如何？能力边界是什么？

#### GLM的预训练任务——Autoregressive Blank infilling

GLM（General Language Model）是“通用”语言模型，这里的“通用”体现在GLM的预训练任务上，也就是***不同粒度的自回归空白填充（autoregressive blank infilling）***，它是对自编码模型（如BERT）、自回归模型（如GPT）以及编解码模型（如T5）三者预训练任务的融合，之前普遍认为这三类模型分别擅长NLU、无条件生成、条件生成，而GLM集三者优势于一体：***在同等数据和模型参数量的情况下，GLM在三个任务类型上均达到最佳***。

如下图所示，首先用[MASK]遮掩掉文本中的若干个span，然后将这些span打乱（相当于做全排列），并按照打乱的顺序依次预测这些span；在预测每个span时，会按照自回归的方式从左往右生成token，此时的位置编码会有两套，一套是当前span对应遮掩字符[MASK]在原文本中的位置，另一套是当前token在span内部的位置；

- span的长度：根据泊松分布（$\lambda = 3$）生成；
- span的个数：重复采样遮掩span，直至15%的文本被遮掩；

![](/static/imgs/glm-fig-1.png)

假设在文本$x=[x_1, x_2, ..., x_n]$上采样$m$个遮掩span $\{s_1, s_2, ..., s_m\}$，其中第$i$个span $s_i = \{s_{i,1}, s_{i,2}..., s_{i, l_i}\}$的长度为$l_i$；$x_{corrupt}$表示在原始文本基础上用[MASK]字符替换所有遮掩span得到的文本；

若$Z_m$是所有可能的遮掩span顺序，则自回归空白填充这个预训练任务可表示为：

<p class="text-center">
$$
\max_{\theta} \Bbb{E}_{z \sim Z_m} \big[ \sum_{i=1}^m \log p_{\theta}(s_{z_i}|x_{corrupt, s_{z \lt i}}) \big] \\
p_{\theta}(s_i|x_{corrupt, s_{z \lt i}}) = \prod_{j=1}^{l_i} p(s_{i,j}|x_{corrupt}, s_{z \lt i}, s_{i,\lt j})
$$
<p>

自回归空白填充是对三种预训练任务的融合，这体现在当遮掩span的数量和长度在取不同值时，自回归空白填充任务可以退化成MLM、条件生成和无条件生成。

但为了让GLM同时具有NLU和NLG的能力，考虑在上述自回归空白填充的基础上，增加以下两个任务：

- 文档级：采样单个span，长度服从均匀分布，覆盖原文档50%-100%长度的内容；
- 句子级：限制遮掩span必须为完整的句子。

GLM的三个预训练任务大体相似，只是遮掩的粒度不同（span的数量和长度）。

#### GLM模型结构

GLM在Transformer的基础上做了以下改动：

- 重新组织layer normalization和残差连接的顺序；
- 使用单个线性层来预测输出token；
- 用GeLU激活函数替换ReLU；
- 两套位置编码，都是learned position embedding，并在输入时和token embedding相加；

---

[1] [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/pdf/2103.10360)

