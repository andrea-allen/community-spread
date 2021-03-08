library(tidyverse)
library(ggplot2)
library(tidyr)
library(RColorBrewer)

devtools::install_github("uclalawcovid19behindbars/behindbarstools")
data <- behindbarstools::read_scrape_data(all_dates = TRUE, coalesce = TRUE)

# get ICE facility data
ice <- data %>% 
  filter(Jurisdiction == 'immigration') %>%
  mutate(covid_rate = Residents.Confirmed / Population.Feb20,
         date = Date,
         county = County,
         state = State) %>%
  select(date, state, county, Facility.ID, Name,
         Residents.Confirmed, Residents.Deaths, Population.Feb20, covid_rate)

ggplot(ice %>% filter(state=='California'), aes(x=date, y=Residents.Confirmed)) +
  geom_line( color="steelblue") + 
  facet_wrap(~ Name)

# get county data
nyt <- read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

joined <- nyt %>%
  left_join(ice, by = c('date', 'state', 'county'))

joined <- joined %>%
  select(date, state, county, Name, cases, Residents.Confirmed) %>%
  pivot_longer(cases:Residents.Confirmed, 
               names_to = "Location", 
               values_to = "Cases") %>%
  mutate(Location = case_when(Location == 'cases' ~ 'County',
                              Location == 'Residents.Confirmed' ~ 'ICE Facility'))

diff <- joined %>%
  group_by(state, county, Name, Location) %>%
  arrange(date) %>%
  mutate(pct_change = (Cases - lag(Cases))/lag(Cases))

ggplot(diff %>% filter(state=='California' & !is.na(Name)),
       aes(x=date, y=pct_change, color=Location)) +
  geom_line() + 
  scale_color_manual(values = c("#293352", "#FC4E07")) +
  facet_wrap(~ Name, scales = 'free_y')

# day over day covid cases in counties vs. facilities
ggsave("community_spread_CA.pdf", width = 14, height = 7)

# covid rate comparision
census <- read_csv('Downloads/co-est2019-alldata.csv') %>%
  select(COUNTY, STNAME, CTYNAME, POPESTIMATE2019)

census$county <- gsub("\\s*\\w*$", "", census$CTYNAME)
census$state <- census$STNAME
census$population <- census$POPESTIMATE2019

census <- census %>% select(county, state, population)

nyt2 <- nyt %>%
  left_join(census, by = c('county', 'state')) %>%
  mutate(county_covid_rate = cases/population)

joined2 <- nyt2 %>%
  left_join(ice, by = c('date', 'state', 'county'))

joined2 <- joined2 %>%
  select(date, state, county, Name, covid_rate, county_covid_rate) %>%
  pivot_longer(covid_rate:county_covid_rate, 
               names_to = "Location", 
               values_to = "Case Rate") %>%
  mutate(Location = case_when(Location == 'covid_rate' ~ 'ICE Facility',
                              Location == 'county_covid_rate' ~ 'County'))

ggplot(joined2 %>% filter(state=='California' & !is.na(Name)),
       aes(x=date, y=`Case Rate`, color=Location)) +
  geom_line() + 
  scale_color_manual(values = c("#293352", "#FC4E07")) +
  facet_wrap(~ Name, scales = 'free_y')

# covid rates in counties vs. facilities
ggsave("community_spread_rate_CA.pdf", width = 14, height = 7)
