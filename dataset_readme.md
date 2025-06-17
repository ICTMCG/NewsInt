# 🗞️ NewsInt Dataset

This dataset is released as part of the paper: **"Exploring news intent and its application: A theory-driven approach"**  
Published in *Information Processing & Management*, 2025.  

📄 Paper Link: https://arxiv.org/abs/2312.16490

🛠️ Project: https://github.com/ICTMCG/NewsInt

🔗 DOI: https://doi.org/10.1016/j.ipm.2025.104229

---

# 📦 Dataset Structure

```
📦NewsInt
 ┣ 📂news
 ┃ ┣ 📜domain_info_all.json     # Domain-level label of news articles
 ┃ ┣ 📜news_docs.json           # All news documents with full metadata
 ┃ ┣ 📜news_topic.json          # Potential topic assignments for each news article
 ┃ ┣ 📜train.csv                # Training set (news ID and associated labels)
 ┃ ┣ 📜val.csv                  # Validation set (news ID and associated labels)
 ┃ ┗ 📜test.csv                 # Test set (news ID and associated labels)
 ┣ 📂post
 ┃ ┣ 📜post.json                # All social media posts, with reply and tree structure
 ┃ ┣ 📜post_docs.json           # Metadata for all posts
 ┃ ┗ 📜id_post_pair.json        # Mapping between news ID and associated post IDs
 ┗ 📜README.md                  # This file
 ```

## 📰 News Data

Located in `news/`, each news item is uniquely identified by an ID and described with metadata in `news_docs.json`. 

- `domain_info_all.json`: All news domain information with labels featuring disinformation, satire, propaganda, science_level, bias, and factuality.
- `news_docs.json`: All news documents with full metadata, identified by a news ID (dict.keys()).
- `news_topic.json`: Potential topic assignments for each news article, identified by a news ID.
- `*.csv`: The dataset is split into `train.csv`, `val.csv`, and `test.csv` for supervised tasks.

An example entry of `news_docs.json`:
```
"12149": {
  "domain": "cnet.com",
  "date": "2020-03-12",
  "author": ["Steven Musil"],
  "title": "Coronavirus testing will be free for all Americans, CDC director says",
  "content": "The commitment comes during a back-and-forth with a member of Congress who...",
  "url": "https://www.cnet.com/news/coronavirus-testing-will-be-free..."
}
```

## 💬 Social Media Posts

The `post/` directory contains comments and discussions referencing the news articles. Each post is identified by a post ID (pid). Reply relationships (if any) are represented as parent-child threads, indicating the iterative structure of discussions.

- `id_post_pair.json`: Maps news IDs to their associated post IDs.
- `post.json`: Contains auxiliary information for each post.
- `post_docs.json`: Contains meta information for each post.

An example entry of `post.json`:
```
{
  "pid": "gpblmvf",
  "parent": "None",
  "child": ["gpbyv7w", "gpbnu8y"],
  "date": "2021-03-01 18:53:14",
  "news": {
    "26006": {
      "domain": "marketwatch.com",
      "url": "https://www.marketwatch.com/..."
    }
  },
  "content": "what an ignorant take...",
  "subreddit": "r/politics",
  "user": "1463ac30..."
}
```

# 📖 Citation
If you find this dataset useful, please cite our paper:

```
@article{WANG2025104229,
  title = {Exploring news intent and its application: A theory-driven approach},
  journal = {Information Processing & Management},
  volume = {62},
  number = {6},
  pages = {104229},
  year = {2025},
  doi = {https://doi.org/10.1016/j.ipm.2025.104229},
  author = {Zhengjia Wang and Danding Wang and Qiang Sheng and Juan Cao and Siyuan Ma and Haonan Cheng}
}
```
