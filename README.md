# News Creation Intent Recognition

<div align= center>
<img src="figs/fig1.png" width="600px" >
</div>

This repo contains the official dataset **NewsInt** for news intent recognition of the research paper "**Exploring news intent and its application: A theory-driven approach**" [[pdf](https://arxiv.org/abs/2312.16490)].

📦 **NewsInt Dataset**: a fine-grained labeled dataset for news creation intent recognition. [[dataset_readme](https://github.com/ICTMCG/NewsInt/blob/main/dataset_readme.md)]

---

<!-- ⚒️ **DMInt Method**: a plug-in method for news intent application -->

## 1. Conceptual Deconstruction-based News INTent Understanding Framework (NINT)

<div align= center>
<img src="figs/fig2.png" width="700px" >
</div>

Intent refers to a cognitive state that emerges from rational planning, grounded in the agent's desires and beliefs. News creation intent refers to the purpose or intention behind the creation of a news article. 

We present the first **Conceptual Deconstruction-based News INTent Understanding framework (NINT)**, deconstructing news intent from perspectives of rational action and outcomes. 

NINT deconstructs the concept of news intent into beliefs, desires, and plans using interdisciplinary theories. It further situates these elements within the specific context of news to investigate the concrete manifestations of news intent.

**To the best of our knowledge, this is the first news creation intent dataset through a deconstruction approach.**


## 2. Dataset Construction

The overall process of building the NewsInt dataset is shown below:

<div align= center>
<img src="figs/fig3.png" width="500px" >
</div>

We collect raw data from 511 news domains, resulting in 12,959 news articles with an average of 5.45 discussion posts from Reddit for each news article. The obtained news articles show diverse distribution on contemporary topics (such as general politics or the US presidential race, the COVID-19 pandemic, women’s and men’s rights, climate change, vaccines, abortion, gun control, 5G, etc.).


Data distribution in factuality (left) and political bias level (right):

<div align= center>
<img src="figs/fig4.png" width="500px" >
</div>

An Instance from the NewsInt dataset:

<div align= center>
<img src="figs/fig5.png" width="500px" >
</div>

For more details on data collection, annotation, and dataset analysis, please refer to our original paper "**Exploring news intent and its application: A theory-driven approach**".

## 3. Usage

Please fill out this form: [Application to Use the Dataset NewsInt](https://forms.office.com/r/9QvFCFT835) to request access. [[dataset_readme](https://github.com/ICTMCG/NewsInt/blob/main/dataset_readme.md)]

---
If you find our paper useful, please cite:

```
@article{WANG2025104229,
  title = {Exploring news intent and its application: A theory-driven approach},
  journal = {Information Processing & Management},
  volume = {62},
  number = {6},
  pages = {104229},
  year = {2025},
  issn = {0306-4573},
  doi = {https://doi.org/10.1016/j.ipm.2025.104229},
  url = {https://www.sciencedirect.com/science/article/pii/S0306457325001700},
  author = {Zhengjia Wang and Danding Wang and Qiang Sheng and Juan Cao and Siyuan Ma and Haonan Cheng},
  keywords = {Intent mining, News media, Social computing},
  abstract = {Understanding the intent behind information is crucial. However, news as a medium of public discourse still lacks a structured investigation of perceived news intent and its application. To advance this field, this paper reviews interdisciplinary studies on intentional action and introduces a conceptual deconstruction-based news intent understanding framework (Nint). This framework identifies the components of intent, facilitating a structured representation of news intent and its applications. Building upon Nint, we contribute a new intent perception dataset. Moreover, we investigate the potential of intent assistance on news-related tasks, such as significant improvement (+2.2% macF1) in the task of fake news detection. We hope that our findings will provide valuable insights into action-based intent cognition and computational social science.}
}
```
