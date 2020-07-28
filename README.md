# Predicting Flight Delays Using Historical Data

## Objective: 

Have you ever been close to booking an important flight, and wondering whether you will make your connecting flight? Wouldn't it be great to know what the probability of your first flight getting delayed would be, so that you can better plan your layover, and avoid missing your flights? 

That's what this project aim to fix. By using historical flight data, I hope to be able to create a machine learning tool that can help passengers when booking flights, or even airline companies and travel agencies, to reduce the risk of missing a connection. 

## Executive Summary

It was possible to train a model that was better than the baseline, using the historical features alone presented in the data from the Department of Transportation website. However, given the large volume of the data, the models were only able to achieve around 60% accuracy / 40% recall on test data. Even after adding weather data, the model did not improve significantly. From the exploration that come out of the data, I suspect that the biggest missing piece to help improve the model was the likelihood that the incoming plan was delayed. If possible, I believe that this would greatly help improve the model.

## Data 

The data used for this project comes from the Department of Transportation website, which stores flights on-time performance fro 1987 to present. You find find the website link [here](https://www.kaggle.com/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)

For the scope of the project, we will only be looking at all flights in 2018, to make the data size more managable. Additionally, we will only be exploring the patterns of the top 15 busiest airports in the US. Busiest airport was defined based on the "total passenger boarding". The list can be found [here](https://en.wikipedia.org/wiki/List_of_the_busiest_airports_in_the_United_States)

All major domestic US airlines are included in the dataset, and we will look to use them all for exploration, as well as modeling.

## Hypothesis:

* $H_0$: Historical flight delays information is not a helpful feature that can help predict probability of a fligth being delayed.
* $H_a$: Historical flight delays information is a helpful feature that can help predict probability of a flight being delayed.



## Project Phases

The project is broken into two phases. The first phase will look to explore the data, and create a model just using the information provided by the Department of Transportation website. The idea is to be able to develop features that can help the model predict whether a specific airline, or a particular airport, is more prone to experience delays versus others. 

The second phase of the project will look to incorporate weather data, in addition to the information provided by the Department of Transportation. As described in the research paper by S Borsky and C Unterberger$^1$, weather can play a signficant role in flight delays. As such, I hope to incorporate hourly weather data into our dataset, to further explore if there is a relationship, and see if it can be useful for modeling. You can find a link to the paper [here](https://www.sciencedirect.com/science/article/pii/S2212012218300753#sec3).

$^1$Borsky, S and Unterberger, C (2019) ‘Bad weather and flight delays: The impact of sudden and slow onset weather events’, Economics of Transportation, Volume 18.  

## Conclusion

I believe that I was able to create a usable model that can help predict the likelihood of a flight getting delayed. While this is only a prototype, and not meant to be used commercially, I do believe that there is strong evidence to support the idea behind this project, and that with further work and access to more computational power, it would be possible to create a strong model that airlines and travel agencies can use to increase performance, and reduce the risk of a customer missing an important connection. 