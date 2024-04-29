#### Setup =======================================================================
source("packages.R")
options(scipen = 999)
global_seed = 123

#### Data =======================================================================

load("../input/data-processed/cross_service_cleaned.RData")

# add id
cross_service <- cross_service %>% 
  mutate(id = row_number())

#### Create Corpus =======================================================================

cross_service_corpus <- cross_service %>%
  corpus(text_field = "text_cleaned")

docnames(cross_service_corpus) <- cross_service$id

docvars(cross_service_corpus)$text <- as.character(cross_service_corpus) 
# assign the actual text as document-level variable to each quasi-sentence

head(docvars(cross_service_corpus))

#### Create and Filter the DFM =======================================================================

# create the dfm
dfm_cross_service <- cross_service_corpus %>% 
  tokens() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  dfm()

# remove all features which occur only once
dfm_cross_service <- dfm_cross_service %>% 
  dfm_trim(min_termfreq = 2)

# only keep non-empty documents
dfm_cross_service <- dfm_subset(dfm_cross_service, ntoken(dfm_cross_service) > 0)

# re-assing doc names and id
new_doc_count <- ndoc(dfm_cross_service)
docnames(dfm_cross_service) <- 1:new_doc_count
docvars(dfm_cross_service, "id") <- 1:new_doc_count

# remove all documents which contain less than 5 tokens
#dfm_cross_service <-  dfm_subset(dfm_cross_service, ntoken(dfm_cross_service) >= 5)

#### Create STM =======================================================================

stm_cross_service <- convert(dfm_cross_service, 
                       to = "stm")

saveRDS(stm_cross_service, file="../input/models/stm_cross_service.RData")


#### Explore Different Ks =======================================================================

parallel::detectCores() 

K = c(20, 30, 40, 50)

set.seed(global_seed)
fit <- searchK(stm_cross_service$documents, 
               stm_cross_service$vocab,
               data = stm_cross_service$meta,
               prevalence =~ moderation_outcome + minority_group + dataset,
               K = K,
               cores = 1,
               verbose = TRUE)

saveRDS(fit, file="../input/models/cross_service-fit.RData")

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
    axis.title = element_text(size = 22),
    axis.text = element_text(size = 18),
    plot.title = element_text(size = 18, face = "bold"),
    strip.text = element_text(size = 18, face = "bold")
  )

statistical_fit

ggsave(plot = statistical_fit,
       filename = "../output/cross_service_stm_statistical_fit.png", 
       width = 10, 
       height = 6,
       dpi = 300)

#### Run Best K Model =======================================================================

topic_model_30 <- stm(documents = stm_cross_service$documents, 
                      vocab = stm_cross_service$vocab,
                      data = stm_cross_service$meta,
                      K = 30, 
                      prevalence =~ moderation_outcome + minority_group + dataset,
                      max.em.its = 75,
                      init.type = "Spectral",
                      verbose = TRUE)

saveRDS(topic_model_30, file="../input/models/cross_service_topic_model_30.RData")

topic_model_50 <- stm(documents = stm_cross_service$documents, 
                      vocab = stm_cross_service$vocab,
                      data = stm_cross_service$meta,
                      K = 50, 
                      prevalence =~ moderation_outcome + minority_group + dataset,
                      max.em.its = 75,
                      init.type = "Spectral",
                      verbose = TRUE)

saveRDS(topic_model_50, file="../input/models/cross_service_topic_model_50.RData")
