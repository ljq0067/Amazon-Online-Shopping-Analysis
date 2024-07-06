library(httr)
library(jsonlite)

# open the folder so that it can be written or read
setwd("D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Assignment1/API_R")

# use api to get raw data
response <- GET("https://newsapi.org/v2/everything?q=online-shopping&apiKey=237d79255be746ca8b2a73cddddac68f")
# response

# use json
# toJSON(fromJSON(content(response, as="text")), pretty = TRUE)
rawdata <- fromJSON(content(response, as="text"))
# rawdata

# select needed information
publishedat <- rawdata$articles$publishedAt
author <- rawdata$articles$author
title <- rawdata$articles$title
source_name <- rawdata$articles$source$name
description <- rawdata$articles$description

# create data frame
data <- data.frame(publishedat,author,source_name,title,description,stringsAsFactors = FALSE)

# output
write.csv(data, file = "online-shopping.csv")
