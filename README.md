# Watching the Watchers: A Comparative Audit of Cloud-Based Commercial Content Moderation Services

## Background

### Master Thesis Project

This repository documents the data and computational methods underlying a master thesis in fulfilment of the requirements for a Masters in Data Science for Public Policy (2024) at the Hertie School in Berlin. The thesis was written in partnership with the **Weizenbaum Institute Berlin**, more specifically the working group on Data, Algorithmic Systems and Ethics, and supervised by **Prof. Simon Munzert** from Hertie.

Feel free to contact me under a.oueslati@students.hertie-school.org for any further requests.

### Short Summary

The paper conducts a comparative audit of four major commercial cloud-based content moderation services, offered by Amazon, Google, Microsoft and OpenAI. The paper's **contribution** is twofold. First, it offers the first comprehensive independent assessment of these algorithms, which are likely not only in use at the companies themselves, but also deployed by a range of smaller organisations through the SaaS model. Second, the paper advances a suite of tests which may inform future black-box third-party audits of content moderation algorithms. To these ends, the paper implements **three experiments**. First, we evaluate service performance on three popular hate speech datasets, ToxiGen[^1], Jigsaw[^2] and MegaSpeech[^3], which were chosen for their popularity and capacity to capture different forms of hate speech. We compute performance metrics both at the aggregate- and at the group-level, parting from most prior research by extending the group-level analysis to a total of eight minority identities. Second, we further test for group-specific biases by running Perturbation Sensitivity Analyses on the most common identity tokens associated with each of the eight minority groups. In essence, we measure the extent to which content moderation services attach a negative bias to these minority tokens, compared to counterfactual majority tokens. We construct these examples from the Identity Phrase Templates in Dixon et al. (2018) and MegaSpeech.[^4] Third, we deploy a Structural Topic Model to explore substantively which topics characterize phrases that were either over- or undermoderated by all services, compared to correctly moderated phrases.[^5]

For further please details, including further background, methodology and results, please also see the paper attached to this repository.

## Repository Overview

Below you find a brief summary of the main folders included in this repository and their link to the analyses from the paper.

### IdentityExtractionLSTM

This folder contains the scripts to train and deploy a BiLSTM to assign identity labels to MegaSpeech. The classifier was trained exploiting a relevant dataset by Yoder et al. (2022), achieving an accuracy of 78% on a hold-out test dataset.[^6]

### API-Pipeline

This folder includes all scripts to call the content moderation APIs. Each API is called by a custom script, including the configurations and parallelisation for that specific API, and then integrated in a joint script to call all APIs in parallel. Further, the folder entail all pre-processing scripts, both in relation to Experiment 1 (ToxiGen, Jigsaw and MegaSpeech), and Experiment 2 (Identity Phrase Templates and MegaSpeech). Lastly, it contains the scripts to compute all performance metrics in relation to Experiments 1 and 2, as well as their appropriate visualisations. All scripts are written in Python.

### Structural-Topic-Model

This folder exclusively pertains to Experiment 3 and is entirely written in R. The switch in programming language is primarily motivated by the superb R stm package from Roberts et al. (2019).[^5] Scripts cover pre-processing, the exploration and comparison of varying topic values K, as well as ultimate analysis and visualisation. While the paper only reports Structural Topic Model results for those cases which were moderated consistently across all moderation services, the repository also includes scripts to explore substantive moderation patterns for Google Content Moderation. The results for Google were excluded from the final paper for reasons of brevity.

### Note on Reproducability

API keys were invalidated to avoid abuse. However, if evaulators' of this thesis want to replicate particular results, this can be accomodated, ideally on a smaller sample. Extensive reproduction would also be feasible, but requires prior discussion. Running the involved API calls does not require any GPU support, parallelisation and batching is embedded in the call scripts. The BiLSTM model was executed with GPU support (T4, 15GB) to reduce run time.

Given GitHub's data constraints, the pre-trained classification model was excluded from the repository. It can be made available upon request. Further, the raw dataset for Jigsaw exceeds the GitHub data limit, but can be accessed directly via [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). 

## References
[^1]: THartvigsen, T., Gabriel, S., Palangi, H., Sap, M., Ray, D. and Kamar, E. (2022), Toxi-Gen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection, in ‘60th Annual Meeting of the Association for Computational Linguistics’.
[^2]: JJigsaw (2019), ‘Jigsaw toxic comment classification challenge’. Last accessed 2024-04-29. URL: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge.
[^3]: Pendzel, S., Wullach, T., Adler, A. and Minkov, E. (2023), ‘Generative AI for Hate Speech Detection: Evaluation and Findings’, arXiv preprint arXiv:2311.09993.
[^4]: Dixon, L., Li, J., Sorensen, J., Thain, N. and Vasserman, L. (2018), Measuring and Mitigating Unintended Bias in Text Classification, in ‘Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society’, AIES ’18, Association for Computing Machinery, New York, NY, USA, pp. 67–73.
[^5]: Roberts, M., Stewart, B. and Tingley, D. (2019), ‘stm : An R Package for Structural Topic Models’, Journal of Statistical Software 91.
[^6]: Yoder, M. M., Ng, L. H. X., Brown, D. W. and Carley, K. M. (2022), ‘How hate speech varies by target identity: A computational analysis’, arXiv preprint arXiv:2210.10839.
