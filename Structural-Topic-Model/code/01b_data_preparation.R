#### Setup =======================================================================

source("packages.R")
options(scipen = 999)

#### Data =======================================================================

# read token data
token <- read_excel("../../API-Pipeline/input/psa/token_conversion.xlsx")

# read data
cross_service <- read.csv("../input/data-raw/cross_service_consistent_moderation_stm_input.csv")

# remove once all labels are consistent
group_filter <- c("black", "latinx", "lgbtq", "asian", "jewish", "female", "disability")
cross_service <- cross_service %>% 
  filter(minority_group %in% group_filter)

# balance by dataset

cross_service %>% 
  group_by(dataset) %>% 
  summarise(n=n())


# get smallest N per dataset and sample for balance across datasets 
smallest_n_dataset <- cross_service %>% 
  group_by(dataset) %>% 
  summarise(n=n()) %>%
  summarise(min(n)) %>% 
  pull

# set.seed(global_seed)
# cross_service <- cross_service %>% 
#  group_by(dataset) %>% 
#  sample_n((smallest_n_dataset)) %>% 
#  ungroup


#### Examine the Distribution =======================================================================

# group
group_stats <- cross_service %>%
  group_by(minority_group, moderation_outcome) %>%
  summarise(count = n()) %>%
  mutate(total_count = sum(count)) %>%
  group_by(minority_group) %>%
  mutate(
    share_overmoderated = count[moderation_outcome == "overmoderated"] / total_count,
    share_undermoderated = count[moderation_outcome == "undermoderated"] / total_count
  ) %>% 
  dplyr::select(minority_group, total_count, share_undermoderated, share_overmoderated) %>% 
  mutate(level = "group") %>% 
  slice(1)


# dataset
dataset_stats <- cross_service %>%
  group_by(dataset, moderation_outcome) %>%
  summarise(count = n()) %>%
  mutate(total_count = sum(count)) %>%
  group_by(dataset) %>% 
  mutate(
    share_overmoderated = count[moderation_outcome == "overmoderated"] / total_count,
    share_undermoderated = count[moderation_outcome == "undermoderated"] / total_count
  ) %>% 
  dplyr::select(dataset, total_count, share_undermoderated, share_overmoderated) %>% 
  mutate(level = "dataset") %>% 
  slice(1) 

# Append the two data frames together
combined_stats <- rbind(group_stats, dataset_stats)

#### Text Pre-Processing =======================================================================

# vector containing identity tokens
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
    str_replace_all("[^\x01-\x7F]", "") %>% 
    # remove numeric values
    str_remove_all("\\d+") %>% 
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
cross_service <- cross_service %>%
  mutate(text_cleaned = clean_text(text))

# remove empty text (after cleaning)
cross_service <- cross_service %>% 
  filter(!(text_cleaned == "")) %>% 
  distinct(text_cleaned, .keep_all = T)

# show random examples of pre and post cleaning
cross_service %>%
  as_tibble() %>%
  slice_sample(n = 10) %>%
  select(text, text_cleaned) %>%
  kable("html") %>%
  kable_styling(full_width = TRUE) %>%
  column_spec(1:2, width = "50%")

# remove empty text (after cleaning)
cross_service <- cross_service %>% 
  filter(!(text_cleaned == ""))

# save pre-processed data
save(cross_service, 
     file = "../input/data-processed/cross_service_cleaned.RData")
