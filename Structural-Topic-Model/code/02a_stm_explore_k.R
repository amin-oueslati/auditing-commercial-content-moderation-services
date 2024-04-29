#### Setup =======================================================================
source("packages.R")
options(scipen = 999)
global_seed = 123

#### Data =======================================================================

load("../input/data-processed/google_cleaned.RData")

# add id
google <- google %>% 
  mutate(id = row_number())

#### Create Corpus =======================================================================

google_corpus <- google %>%
  corpus(text_field = "text_cleaned")

docnames(google_corpus) <- google$id

docvars(google_corpus)$text <- as.character(google_corpus) 
# assign the actual text as document-level variable to each quasi-sentence

head(docvars(google_corpus))

#### Create and Filter the DFM =======================================================================

# create the dfm
dfm_google <- google_corpus %>% 
  tokens() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  dfm()

# remove all features which occur only once
dfm_google <- dfm_google %>% 
  dfm_trim(min_termfreq = 2)

# only keep non-empty documents
dfm_google <- dfm_subset(dfm_google, ntoken(dfm_google) > 0)

# re-assing doc names and id
new_doc_count <- ndoc(dfm_google)
docnames(dfm_google) <- 1:new_doc_count
docvars(dfm_google, "id") <- 1:new_doc_count

# remove all documents which contain less than 5 tokens
# dfm_google <-  dfm_subset(dfm_google, ntoken(dfm_google) >= 5)

#### Create STM =======================================================================

stm_google <- convert(dfm_google, 
                       to = "stm")

saveRDS(stm_google, file="../input/models/stm_google.RData")
  
#### Explore Different Ks =======================================================================

parallel::detectCores() 

K = c(20, 30, 40, 50)

set.seed(global_seed)
fit <- searchK(stm_google$documents, 
               stm_google$vocab,
               data = stm_google$meta,
               prevalence =~ moderation_outcome*minority_group + dataset,
               K = K,
               cores = 1,
               verbose = TRUE)

saveRDS(fit, file="../input/models/google-fit.RData")

#### Statistical Fit =======================================================================

statistical_fit <- fit$results %>%
  pivot_longer(cols = -K, names_to = "metric", values_to = "value") %>% 
  filter(metric %in% c("lbound", "exclus", "residual", "semcoh")) %>%
  mutate(value = map_dbl(value, 1)) %>% 
  mutate(K = map_dbl(K, 1))

statistical_fit <- ggplot(statistical_fit, aes(x = K, y = value, color = metric)) +
  geom_point() + geom_line(linewidth = 1.5) +
  geom_vline(aes(xintercept = 20) , alpha = .5, linewidth = 0.5) +
  geom_vline(aes(xintercept = 30) , alpha = .5, linewidth = 0.5) +
  geom_vline(aes(xintercept = 40) , alpha = .5, linewidth = 0.5) +
  geom_vline(aes(xintercept = 50) , alpha = .5, linewidth = 0.5) +
  scale_x_continuous(breaks = c(20, 30, 40, 50)) +
  guides(color = "none") +
  facet_wrap(~metric, scales = "free", 
             labeller = labeller(metric = c(exclus = "Exclusivity", 
                                            lbound = "Variational Lower Bound", 
                                            residual = "Residual", 
                                            semcoh = "Semantic Coherence"))) +
  labs(y = NULL) +
  theme_pubr() +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    plot.title = element_text(size = 16, face = "bold"),
    strip.text = element_text(size = 14, face = "bold")
  )

statistical_fit

ggsave(plot = statistical_fit,
       filename = "../output/google_stm_statistical_fit.png", 
       width = 10, 
       height = 6)

#### Run Best K Model =======================================================================

set.seed(global_seed)
topic_model_30 <- stm(documents = stm_google$documents, 
                      vocab = stm_google$vocab,
                      data = stm_google$meta,
                      K = 30, 
                      prevalence =~ moderation_outcome*minority_group + dataset,
                      max.em.its = 75,
                      init.type = "Spectral",
                      verbose = TRUE)

saveRDS(topic_model_30, file="../input/models/google_topic_model_30.RData")

topic_model_50 <- stm(documents = stm_google$documents, 
                      vocab = stm_google$vocab,
                      data = stm_google$meta,
                      K = 50, 
                      prevalence =~ moderation_outcome*minority_group + dataset,
                      max.em.its = 75,
                      init.type = "Spectral",
                      verbose = TRUE)

saveRDS(topic_model_50, file="../input/models/google_topic_model_50.RData")
