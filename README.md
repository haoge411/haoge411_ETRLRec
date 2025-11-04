# ETRLRec: Enhancing Temporal Reasoning by Reinforcement Learning for LLM-based Sequential Recommendation
## Abstract
Sequential recommendation (SR) aims to predict user's next interaction by analyzing interaction history. Recently, large language models (LLMs) have shown increasing potential in SR tasks. Unlike conventional SR models that rely on interaction information, LLMs can use broader external knowledge for SR tasks. Due to the sequential dependencies and dynamic user preferences in interaction history, SR tasks exhibit strong temporal dynamics. Therefore, SR models require temporal modeling capabilities or temporal knowledge for improving SR performance. However, LLMs lack such. temporal knowledge, limiting their SR performance. 

Researchers have tried to take temporal knowledge into consideration with LLMs for SR tasks. They integrate either temporal knowledge embeddings or text with LLMs, which improves SR performance to some extent. However, temporal knowledge embedding causes relationship confusion, while temporal knowledge text leads to information loss, limiting the performance. And most of these methods are trained solely on supervised fine-tuning (SFT), resulting in poor generalization and performance ceiling. Although reinforcement learning (RL) can alleviate these SFT issues, most RL methods typically define reward module simply as "whether the next interaction is match", which contains limited valuable recommendation information, leading to suboptimal reward signals.

To address aforementioned issues, we propose a novel framework, Enhancing Temporal Reasoning by Reinforcement Learning for LLM-based Sequential Recommendation (ETRLRec). ETRLRec first utilizes temporal knowledge to guide LLM in constructing temporal reasoning, and then it distills the constructed temporal reasoning to a smaller and more cost-effective LLM. These two stages enable LLM to acquire temporal reasoning ability, allowing it to analyze and solve SR tasks in a temporal dynamic pattern. In addition, to address issues of SFT and overly simplistic RL reward module, ETRLRec proposes a novel RL-based training method with an elaborate reward module to further improve SR performance. Extensive experiments on three real datasets validate ETRLRec's effectiveness.

## Rough View
![delrec_rough](https://github.com/user-attachments/assets/b61bf4fd-9775-4bd5-9e64-b23829873450)

## Paper
DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation ([DELRec.pdf](https://github.com/user-attachments/files/17808804/DELRec.pdf))


## Preparation
1. **Prepare the environment:**
   To install the dependencies for this project, run the following command:
    ```bash
    git clone https://github.com/haoge6660101/DELRec_hao.git
    cd DELRec
    pip install -r requirements.txt
    ```

2. **Prepare the pre-trained Huggingface model of Flan-T5-XL:**
   
    Flan-T5-XL: https://huggingface.co/google/flan-t5-xl

3. **Download the datasets:**

    MovieLens-100K: https://grouplens.org/datasets/movielens/100k/
   
    Steam: https://huggingface.co/datasets/joyliao7777/LLaRA/tree/main/steam     

    Beauty: https://github.com/RUCAIBox/RecSysDatasets

    Home and Kitchen: https://github.com/RUCAIBox/RecSysDatasets
