library(tidyverse)
library(tidycensus)
census_api_key("c04ebbbf45c0cb8155bbf2e821a1421a08fdfa72", install=TRUE)

varNames = load_variables(2024, "acs5", cache =TRUE)
write.csv(varNames, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/acs_var_names.csv")

age_vars = varNames %>% filter(str_detect(name, 'B01001_'))
write.csv(age_vars, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/ageVars_acs.csv")

race_vars = varNames %>% filter(str_detect(name, 'B02001_'))
write.csv(race_vars, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/raceVars_acs.csv")

enrollment_vars = varNames %>% filter(str_detect(name, 'B14001_|B14002_|B14003_|B14006_|B14007'))
write.csv(enrollment_vars, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/enrollmentVars_acs_extended.csv")

poverty_vars = varNames %>% filter(str_detect(name, "B17009_002|B17009_019"))
write.csv(poverty_vars, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/povertyVars_acs.csv")

education_vars = varNames %>% filter(str_detect(name, 'B15001_|B15003_|B150013_'))
write.csv(education_vars, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/educationVars_acs.csv")

language_vars = varNames %>% filter(str_detect(name, 'B16002_'))
write.csv(language_vars, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/languageVars_acs.csv")

disability_vars = varNames %>% filter(str_detect(name, 'B18101_'))
write.csv(disability_vars, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/disabilityVars_acs.csv")

healthInsur_vars = varNames %>% filter(str_detect(name, 'B27001_'))
write.csv(healthInsur_vars, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/healthInsurVars_acs.csv")

get_data_and_join = function(var_list){
  df =  get_acs(geography='county', state="TX", year=2024, variables= c(var_list$name), 
                geometery=TRUE, keep_geo_vars = TRUE)
  
  return(df)
}

tx_county_age = get_data_and_join(age_vars)
write.csv(tx_county_age, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/tx_county_agePop_24.csv")

tx_county_race = get_data_and_join(race_vars)
write.csv(tx_county_race, file="C:/Users/19037/Documents/Measels_Vaccination/tx_county_racePop_24.csv")

tx_county_enrollment = get_data_and_join(enrollment_vars)
write.csv(tx_county_enrollment, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/tx_county_enrollment_24_extended.csv")

tx_county_edu = get_data_and_join(education_vars)
write.csv(tx_county_edu, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/tx_county_education_24.csv")

#tx_county_age = get_acs(geography='county', state="TX", year=2024, variables= c(age_vars$name), 
#                        geometery=TRUE, keep_geo_vars = TRUE) 

tx_county_poverty = get_acs(geography='county', state="TX", year=2024, variables = c("B17009_002", "B17009_019"), 
                                    geometery=TRUE, keep_geo_vars = TRUE)
write.csv(tx_county_poverty, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/tx_county_poverty_24.csv")

tx_county_language = get_data_and_join(language_vars)
write.csv(tx_county_language, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/tx_county_language_24.csv")

tx_county_disability = get_data_and_join(disability_vars)
write.csv(tx_county_disability, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/tx_county_disability_24.csv")

tx_county_healthInsur = get_data_and_join(healthInsur_vars)
write.csv(tx_county_healthInsur, file="C:/Users/19037/Documents/Measels_Vaccination/acs_data/tx_county_health_insurance_24.csv")

#zipUS_25 = get_acs(geography="zcta", variables="B01003_001", year=2025)