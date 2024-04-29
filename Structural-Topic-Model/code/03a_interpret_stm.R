#### Setup =======================================================================
source("packages.R")
options(scipen = 999)
global_seed = 123

#### Load Models =======================================================================

topic_model_30 <- readRDS("../input/models/google_topic_model_30.RData")
topic_model_50 <- readRDS("../input/models/google_topic_model_50.RData")
stm_google <- readRDS("../input/models/stm_google.RData")
load("../input/data-processed/google_cleaned.RData")

#### exclusive words ####

set.seed(global_seed)
label_topics_30 <- labelTopics(topic_model_30, n = 10, frexweight = 0.8)$frex
label_topics_30 <- apply(label_topics_30, 1, function(row) paste(row, collapse = ", "))
label_topics_30 <- cbind(data.frame(Topic = 1:30), label_topics_30)

colnames(label_topics_30) <- c("Topic", "Labels")

label_topics_30 %>% 
  kable(format = "html") %>%
  add_header_above(
    c("Topic Labels (K30)" = 2)) %>%
  kable_styling(bootstrap_options = c("striped", "hover"), 
                font_size = 14,
                full_width = TRUE)

label_topics_50 <- labelTopics(topic_model_50, frexweight = 0.8)$frex
label_topics_50 <- apply(label_topics_50, 1, function(row) paste(row, collapse = ", "))
label_topics_50 <- cbind(data.frame(Topic = 1:50), label_topics_50)

colnames(label_topics_50) <- c("Topic", "Label")

set.seed(global_seed)
label_topics_30 <- labelTopics(topic_model_30, n = 7, frexweight = 0.8)$frex
label_topics_30 <- apply(label_topics_30, 1, function(row) paste(row, collapse = ", "))
label_topics_30 <- data.frame(Topic = 1:30, Labels = label_topics_30)

label_topics_50 <- labelTopics(topic_model_50, n = 7, frexweight = 0.8)$frex
label_topics_50 <- apply(label_topics_50, 1, function(row) paste(row, collapse = ", "))
label_topics_50 <- data.frame(Topic = 1:50, Labels = label_topics_50)

# Create a joint table
joint_table <- full_join(label_topics_30, label_topics_50, by = "Topic")

# Replace NAs in Labels_K20 column with empty strings
joint_table <- joint_table %>% 
  mutate(Labels.x = ifelse(is.na(Labels.x), "", Labels.x))

names(joint_table) <- c("Topic", "Labels K30", "Labels K50")

# Print the joint table
kable(joint_table, format = "html") %>%
  kable_styling(bootstrap_options = c("striped", "hover"), 
                font_size = 14,
                full_width = TRUE)

#### substantive fit: key examples ####

# K30

set.seed(global_seed)
texts_30 <- findThoughts(topic_model_30, texts = stm_google$meta$text)

# Initialize an empty data frame
text_30_kable <- tibble(
  Topic = character(),
  Texts = character()
)

# Iterate over each topic
for (topic_name in names(texts_30$docs)) {
  topic_texts <- texts_30$docs[[topic_name]]
  topic_df <- tibble(
    Topic = topic_name,
    Texts = unlist(topic_texts)
  )
  text_30_kable <- bind_rows(text_30_kable, topic_df)
}

label_topics_30_short <- label_topics_30 %>%
  mutate(label_short = str_extract(Labels, "^([^,]+, [^,]+, [^,]+, [^,]+, [^,]+)"))
colnames(label_topics_30_short) <- c("Topic", "Label", "Label_Short")

text_30_kable <- text_30_kable %>% 
  mutate(Topic = as.integer(str_remove(Topic, "Topic "))) %>% 
  left_join(label_topics_30_short %>% dplyr::select(Topic, Label_Short),
            by = "Topic")

text_30_kable %>%
  distinct(Texts, .keep_all = T) %>% 
  kable(format = "html",
        booktabs = TRUE,
        col.names = c("Topics", "Most Representative Document", "Label"),
        escape = FALSE) %>%
  kable_styling(full_width = TRUE)


# K50

set.seed(global_seed)
texts_50 <- findThoughts(topic_model_50, texts = stm_google$meta$text)

# Initialize an empty data frame
text_50_kable <- tibble(
  Topic = character(),
  Texts = character()
)

# Iterate over each topic
for (topic_name in names(texts_50$docs)) {
  topic_texts <- texts_50$docs[[topic_name]]
  topic_df <- tibble(
    Topic = topic_name,
    Texts = unlist(topic_texts)
  )
  text_50_kable <- bind_rows(text_50_kable, topic_df)
}

label_topics_50_short <- label_topics_50 %>%
  mutate(label_short = str_extract(Labels, "^([^,]+, [^,]+, [^,]+, [^,]+, [^,]+)"))
colnames(label_topics_50_short) <- c("Topic", "Label", "Label_Short")

text_50_kable <- text_50_kable %>% 
  mutate(Topic = as.integer(str_remove(Topic, "Topic "))) %>% 
  left_join(label_topics_50_short %>% dplyr::select(Topic, Label_Short),
            by = "Topic")

text_50_kable %>%
  distinct(Texts, .keep_all = T) %>% 
  kable(format = "html",
        booktabs = TRUE,
        col.names = c("Topic", "Most Representative Document", "Labels"),
        escape = FALSE) %>%
  kable_styling(full_width = TRUE)


#### Statistically Estimate Topic-Moderation_Outcome-Minority Relationships =======================================================================

# estimate effects: moderation
set.seed(global_seed)
estimates_moderation <- estimateEffect(1:50 ~ moderation_outcome,
                            topic_model_50,
                            stm_google$meta)

tidy_estimates <- tidytext::tidy(estimates_moderation)

# filter for moderation_outcome
tidy_estimates <- tidy_estimates %>% 
  filter(p.value < 0.05) %>% 
  filter(str_detect(term, "moderation"))

under_moderated <- stminsights::get_effects(estimates = estimates_moderation,
            variable = "moderation_outcome",
            type = "difference",
            cov_val1 = "under_moderated",
            cov_val2 = "correctly_moderated") %>% 
  mutate(keep = difference > 0 & lower > 0,
         moderation_outcome = "over_moderated") %>% 
  filter(keep)

over_moderated <- stminsights::get_effects(estimates = estimates_moderation,
                                          variable = "moderation_outcome",
                                          type = "difference",
                                          cov_val1 = "over_moderated",
                                          cov_val2 = "correctly_moderated") %>% 
  mutate(keep = difference > 0 & lower > 0,
         moderation_outcome = "under_moderated") %>% 
  filter(keep)

false_moderation_prevalence <- rbind(over_moderated, under_moderated)

false_moderation_prevalence <- false_moderation_prevalence %>% 
  mutate(topic = as.integer(topic)) %>% 
  rename(Topic = topic) %>%
  left_join(text_50_kable,
            by = "Topic",
            relationship = "many-to-many")

false_moderation_prevalence <- false_moderation_prevalence %>%
  mutate(order = paste(moderation_outcome, difference))

ordered_data <- false_moderation_prevalence %>%
  distinct(Topic, .keep_all = TRUE) %>%
  arrange(moderation_outcome, difference) %>%
  mutate(Topic = factor(Topic, unique(Topic)))

color_palette <- c("#1F77B4", "#FF7F0E")  # Blue and orange

# Plotting
false_moderation_topic_prevalence <- ggplot(ordered_data, aes(x = difference, 
                         y = Topic,
                         color = moderation_outcome)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0, size = 1.5) +
  scale_color_manual(values = color_palette) +
  scale_x_continuous(limits = c(-0.01, 0.04),
                     breaks = seq(-0.01, 0.04, by = 0.01)) +
  geom_vline(xintercept = 0, linetype = "solid", size = 0.5) +
  labs(x = "Difference in Prevalence (gamma score) to Correct Moderation", 
       y = "Topic",
       color = NULL) +
  theme_pubr() +
  theme(axis.title.x = element_text(size = 22),
        axis.text.y = element_text(size = 22),
        axis.text.x = element_text(size = 24),
        axis.title.y = element_text(size = 22),
        panel.grid.major.y = element_line(color = "gray", size = 0.2),
        panel.grid.major.x = element_line(color = "gray", size = 0.2),
        legend.text = element_text(size = 22)
  )

ggsave(filename = paste0("../output/google_false_moderation_topic_prevalence", ".png"), 
       plot = false_moderation_topic_prevalence, 
       width = 10, height = 6, dpi = 96)
  

#### Topic Prevalence by Moderation Outcome and Minority Group: 50 Topics =======================================================================

gamma_50 = tidy(topic_model_50, matrix = "gamma")

colnames(gamma_50) <- c("Document", "Topic", "Gamma")

meta_data <- stm_google$meta %>%
  select(id, minority_group, moderation_outcome)

gamma_by_minority_50 <- gamma_50 %>% 
  left_join(meta_data, join_by(Document == id),
            relationship = "many-to-many")

label_topics_50_short <- label_topics_50 %>%
  mutate(label_short = str_extract(Labels, "^([^,]+, [^,]+, [^,]+, [^,]+, [^,]+)"))

colnames(label_topics_50_short) <- c("Topic", "Label", "Label_Short")

gamma_by_minority_50 <- gamma_by_minority_50 %>% 
  group_by(minority_group, moderation_outcome, Topic) %>% 
  summarise(gamma = mean(Gamma)) %>% 
  ungroup() %>% 
  left_join(label_topics_50_short, by = "Topic") %>%
  mutate(Topic = paste0("T ", Topic),
         Topic = reorder(Topic, gamma))

# consistent color palette
color_palette_50 <- viridis(50, option = "D")
color_mapping_50 <- setNames(color_palette_50, unique(gamma_by_minority_50$Topic))

# Define the desired order for moderation outcomes
moderation_outcomes_order <- c("under_moderated", "correctly_moderated", "over_moderated")

# Remove 'unknown' from the data and order the moderation outcomes
gamma_by_minority_50 <- gamma_by_minority_50 %>%
  mutate(moderation_outcome = factor(moderation_outcome, levels = moderation_outcomes_order))

# Get a list of unique minority groups from the filtered data
minority_groups <- unique(gamma_by_minority_50$minority_group)

# Create a list of plots, one for each minority group
plots_list_50 <- lapply(minority_groups, function(group) {
  
  group_data <- gamma_by_minority_50 %>%
    filter(minority_group == group) %>%
    group_by(moderation_outcome) %>%
    slice_max(order_by = gamma, n = 5) %>%
    ungroup() %>% 
    mutate(moderation_outcome = as.factor(moderation_outcome),
           Topic_rw = tidytext::reorder_within(Topic, gamma, moderation_outcome),
           Label_Short = str_wrap(Label_Short, width = 22))
  
  max_gamma_rounded <- ceiling(max(group_data$gamma) / 0.05) * 0.05
  
  if (max_gamma_rounded < 0.15) {
    max_gamma_rounded <- 0.15}
  if (max_gamma_rounded == 0.15) {
    breaks <- seq(0, max_gamma_rounded, by = 0.05)} 
  else {
    breaks <- seq(0, max_gamma_rounded, by = 0.1)}
  
  gg <- group_data %>%
    ggplot(aes(x = Topic_rw,
               y = gamma, 
               label = Label_Short, 
               fill = Topic)) +
    geom_col(show.legend = FALSE) +
    geom_text(hjust = 0, nudge_y = 0.005, size = 7.5) +
    coord_flip() +
    scale_y_continuous(expand = c(0, 0), limits = c(0, max_gamma_rounded*1.9), 
                       breaks = breaks) +
    scale_fill_manual(values = color_mapping_50) +
    labs(x = NULL, y = "Prevalence (Gamma Score)") +
    scale_x_reordered() +
    facet_wrap(~moderation_outcome, scales = "free_y", ncol = 3) +
    theme_pubr() +
    theme(axis.title.x = element_text(size = 26),  # Increase x-axis label font siz
          axis.text.y = element_text(size = 24),  # Increase y-axis label font size
          axis.text.x = element_text(size = 24),   # Increase x-axis tick labels font size
          strip.text = element_text(size = 26))   # Increase facet wrap heading font size
  
  # Return the plot
  gg
})
  
# Name the list elements with the minority group names
plots_list_50 <- setNames(plots_list_50, minority_groups)

for (plot_name in names(plots_list_50)) {
  print(plots_list_50[[plot_name]])
}

# If you want to save all the plots to files, you can loop through the named list
for (name in names(plots_list_50)) {
  ggsave(filename = paste0("../output/google_50_topics_prevalence_by_minority_", name, ".png"), 
         plot = plots_list_50[[name]], 
         width = 18, height = 9, dpi = 96)
}
