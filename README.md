# Watching the Watchers: A Comparative Audit of Cloud-Based Commercial Content Moderation Services

## Context 

This repository documents the data and computational methods underlying a master thesis in fulfilment of the requirements for a Masters in Data Science for Public Policy (2024) at the Hertie School in Berlin. The thesis was written in partnership with the Weizenbaum Institute Berlin, more specifically the working group on Data, Algorithmic Systems and Ethics, and supervised by Prof. Simon Munzert from Hertie. 

The paper conducts a comparative audit of four major commercial cloud-based content moderation services, offered by Amazon, Google, Microsoft and OpenAI. The paper's contribution is twofold. First, it offers the first comprehensive external assessment of these algorithms, which are likely not only in use at the companies themselves, but also deployed by range of smaller organisations through the SaaS model. Second, the paper advances a suite of tests which may inform future black-box third-party audits of content moderation algorithhms. To these ends, the paper implements three experiments. First, we evaluate service performance on three popular hate speech datasets, ToxiGen[^1], Jigsaw[^2] and MegaSpeech[^3], which were chosen for their popularity and capacity to capture different forms of hate speech. We compute performance metrics both at the aggregate- and at the group-level, parting from most prior research by extending the group-level analysis to a total of eight minority identities. Second, we further test for group-specific biases by running Perturbation Sensitivity Analyses[^4] on the most common identity tokens associated with each of the eight minority groups. In essence, we measure the extent to which content moderation services attach a negative bias to these minority tokens, compared to counterfactual majority tokens. We construct these examples from the Identity Phrase Templates in Dixon et al. (2018)[^5] and MegaSpeech. Third, we deploy a Structural Topic Model[^6] to explore substantively which topics characterise phrases that were either over- or undermoderated by all services, compared to correctly moderated phrases.

## Reprodu



## Repository Overview


[^1]: Jigsaw. Jigsaw toxic comment classification challenge., 2019. URL: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge.
[^2]: To add line breaks within a footnote, prefix new lines with 2 spaces.
  This is a second line.
