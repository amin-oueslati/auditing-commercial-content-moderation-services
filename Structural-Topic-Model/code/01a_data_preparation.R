#### Setup =======================================================================

source("packages.R")
options(scipen = 999)

#### Data =======================================================================

# read token data
token <- read_excel("../../API-Pipeline/input/psa/token_conversion.xlsx")

# read data
google <- read.csv("../input/data-raw/google_stm_input.csv")

# change jews to jewish
google <- google %>% 
  mutate(minority_group = recode(minority_group, "jews" = "jewish", "latino" = "latinx"))

google %>% 
  group_by(minority_group) %>% 
  summarise(n=n())

google %>% filter(
  dataset == "megaspeech") %>% 
  group_by(minority_group) %>% 
  summarise(n=n())

# remove once all labels are consistent
group_filter <- c("black", "latinx", "lgbtq", "asian", "jewish", "female", "disability")
google <- google %>% 
  filter(minority_group %in% group_filter)

# balance by dataset

google %>% 
  group_by(dataset) %>% 
  summarise(n=n())

# get smallest N per dataset and sample for balance across datasets 
smallest_n_dataset <- google %>% 
  group_by(dataset) %>% 
  summarise(n=n()) %>%
  summarise(min(n)) %>% 
  pull

google <- google %>% 
  group_by(dataset) %>% 
  sample_n(size = smallest_n_dataset) %>% 
  ungroup()

#### Examine the Distribution =======================================================================

# calculate the counts of each moderation outcome within each minority group
moderation_counts <- google %>%
  filter(moderation_outcome != "unknown") %>% # remove any missing moderation outcomes
  group_by(minority_group, moderation_outcome) %>%
  summarise(count = n()) %>%
  mutate(total_count = sum(count)) %>%
  group_by(minority_group) %>%
  mutate(share = count / total_count) %>% 
  ungroup() %>% 
  mutate(moderation_outcome = 
           fct_relevel(moderation_outcome, "over_moderated", "under_moderated", "correctly_moderated"))

# Create stacked bar plot
color_palette <- rev(brewer.pal(n = 3, name = "Blues"))

moderation_distribution_google <- ggplot(moderation_counts, aes(x = minority_group, y = share, fill = moderation_outcome)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = color_palette,
                    labels = function(x) gsub("_", " ", x)) +
  labs(
    # title = "Statements Classified with Google Moderate Text API",
       fill = "Moderation Outcome") +
  geom_text(aes(label = scales::percent(share, accuracy = 1)), 
            position = position_stack(vjust = 0.5), size = 4) +
  geom_text(aes(label = paste0("N=", total_count)), y = 1.1, size = 4, hjust = 0.5) + # Add sample size above the bars
  scale_y_continuous(limits = c(0, 1.1), 
                     breaks = seq(0, 1, by = 0.25),
                     labels = scales::percent) +
  guides(color = "none") +
  theme_pubr() +
  theme(plot.title = element_text(size = 13), # Increased title font size
        legend.title = element_text(size = 12), # Increased legend title font size
        legend.text = element_text(size = 10), # Increased legend text font size
        legend.position = "right",
        axis.title.y = element_blank(), # Remove y-axis label
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

save(moderation_distribution_google,
     file = "../output/moderation_distribution_google.RData")

# save pre-processed data
ggsave(plot = moderation_distribution_google ,
       filename = "../output/moderation_distribution_google.png", 
       width = 10, 
       height = 6)

#### Text Pre-Processing =======================================================================
  
identity_token <- token$minority_token

clean_text <- function(x) {
  identity_token <- paste(identity_token, collapse = "|")
  x %>%
    # remove URLs
    str_remove_all(" ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)") %>%
    # Remove mentions e.g. "@my_account"
    str_remove_all("@[[:alnum:]_]{4,}") %>%
    # remove hashtags
    str_remove_all("#[[:alnum:]_]+") %>%
    # remove emojis
    # str_replace_all(x, "[^\x01-\x7F]", "") %>% 
    # remove numeric values
    # str_remove_all("\\d+") %>% 
    # replace "&" character reference with "and"
    str_replace_all("&amp;", "and") %>%
    # remove puntucation, using a standard character class
    str_remove_all("[[:punct:]]") %>%
    # replace any newline characters with a space
    str_replace_all("\\\n", " ") %>%
    # make everything lowercase
    str_to_lower() %>%
    # remove any trailing whitespace around the text
    str_trim("both") %>% 
    # remove emojis
    str_replace_all("[^\x01-\x7F]", "") %>% 
    # remove words contained identity_token
    str_remove_all(paste0("\\b(", identity_token, ")\\b"))
}


# apply cleaning function to text
google <- google %>%
  mutate(text_cleaned = clean_text(text))

# remove empty text (after cleaning)
google <- google %>% 
  filter(!(text_cleaned == "")) %>% 
  distinct(text_cleaned, .keep_all = T)

# show random examples of pre and post cleaning
google %>%
  as_tibble() %>%
  slice_sample(n = 10) %>%
  select(text, text_cleaned) %>%
  kable("html") %>%
  kable_styling(full_width = TRUE) %>%
  column_spec(1:2, width = "50%")

# save pre-processed data
save(google, 
     file = "../input/data-processed/google_cleaned.RData")
