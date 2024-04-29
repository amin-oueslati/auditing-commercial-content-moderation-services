# Watching the Watchers: A Comparative Audit of Cloud-Based Commercial Content Moderation Services

## Context 

This repository documents the data and computational methods underlying a master thesis in fulfilment of the requirements for a Masters in Data Science for Public Policy (2024) at the Hertie School in Berlin. The thesis was written in partnership with the <strong>Weizenbaum Institute Berlin</strong>, more specifically the working group on Data, Algorithmic Systems and Ethics, and supervised by <strong>Prof. Simon Munzert</strong> from Hertie.

The paper conducts a comparative audit of four major commercial cloud-based content moderation services, offered by Amazon, Google, Microsoft and OpenAI. The paper's <strong>contribution</strong> is twofold. First, it offers the first comprehensive external assessment of these algorithms, which are likely not only in use at the companies themselves, but also deployed by a range of smaller organisations through the SaaS model. Second, the paper advances a suite of tests which may inform future black-box third-party audits of content moderation algorithms. To these ends, the paper implements <strong>three experiments</strong>. First, we evaluate service performance on three popular hate speech datasets, ToxiGen<sup>1</sup>, Jigsaw<sup>2</sup> and MegaSpeech<sup>3</sup>, which were chosen for their popularity and capacity to capture different forms of hate speech. We compute performance metrics both at the aggregate- and at the group-level, parting from most prior research by extending the group-level analysis to a total of eight minority identities. Second, we further test for group-specific biases by running Perturbation Sensitivity Analyses<sup>4</sup> on the most common identity tokens associated with each of the eight minority groups. In essence, we measure the extent to which content moderation services attach a negative bias to these minority tokens, compared to counterfactual majority tokens. We construct these examples from the Identity Phrase Templates in Dixon et al. (2018)<sup>5</sup> and MegaSpeech. Third, we deploy a Structural Topic Model<sup>6</sup> to explore substantively which topics characterize phrases that were either over- or undermoderated by all services, compared to correctly moderated phrases.

For further please details, including further background, methodology and results, please also see the paper attached to this repository.

## Repository Overview

Below you find a brief summary of the main folders included in this repository and their link to the analyses from the paper.

### API-Pipeline

This folder includes all scripts to call the content moderation services' APIs. Each API is called by a custom script, including the configurations and parallelisation for that specific API, and then integrated in a joint script to call all APIs in parallel. Further, the folder includes all data pre-processing, both in relation to Experiment 1 (ToxiGen, Jigsaw and MegaSpeech), and Experiment 2 (Identity Phrase Templates and MegaSpeech). Lastly, it contains the scripts to compute all performance metrics put forward by the paper and visualise them appropriately. All scripts are written in Python.


### API-Pipeline

This folder includes all scripts to call the content moderation services' APIs. Each API is called by a custom script, including the configurations and parallelisation for that specific API, and then integrated in a joint script to call all APIs in parallel. Further, the folder includes all data pre-processing, both in relation to Experiment 1 (ToxiGen, Jigsaw and MegaSpeech), and Experiment 2 (Identity Phrase Templates and MegaSpeech). Lastly, it contains the scripts to compute all performance metrics put forward by the paper and visualise them appropriately. All scripts are written in Python.

### Structural-Topic-Model

This folder exclusively pertains to Experiment 3 and is entirely written in R. The switch in programming language is primarily motivated by the superb R stm package. Scripts cover pre-processing, the exploration and comparison of varying numbers of topics K, as well as ultimate analysis and visualisation. While the paper only reports Structural Topic Model results for those cases which were moderated consistently across all moderation services, the repository also includes scripts to explore substantive moderation patterns for Google Content Moderation. The results for Google were excluded from the final paper for reasons of brevity.

### Note on Reproducability



## References
[^1]: Jigsaw. Jigsaw toxic comment classification challenge., 2019. URL: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge.
[^2]: 
