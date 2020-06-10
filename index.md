# Introduction 

This project involves building a neural network model to identify ingredients from german recipes. Given the receipe, the named entity recognition model identifies the ingredients from the recipe. 
Open data source with german recipes crawled from chefkoch.de is used for the purpose. 

# Project 

Python pandas library is to read the recipes file. Here is a small snippet of the data. 

``` data_frames.head()
```
![df](data_frame_receipe.PNG) 

```df.shape
``` 
(12190, 8)
We see that there are a total of 12190 recipes, and we split it into train test to build and test the model. 
Recipes from 1 to 11.000 are used to train our model and remaining 1190 are used to test the model. 
We see that our data has Ingredients and instruction column. 
```data_frames.ingredients[4] 
``` 
['250 g Kohlrabi',
 '150 ml Gemüsebrühe',
 '150 ml Milch',
 '250 g Farfalle oder Kelche',
 '100 g Kochschinken',
 '100 g Frischkäse ( evtl. mit Kräutern )',
 'Salz und Pfeffer',
 'einige Stiele Petersilie',
 'evtl. Saucenbinder, hell']
 
 ``` data_frames.Instructions[4]
``` 
'Kohlrabi schälen und klein würfeln. Mit der Brühe und der Milch aufkochen lassen und zugedeckt kochen, bis er gar, aber noch bissfest ist.Währenddessen die Nudeln in reichlich Salzwasser bissfest kochen.Den Kochschinken in kurze Streifen schneiden. Petersilie waschen, trocken tupfen und fein hacken. Alles mit dem Frischkäse unter die Kohlrabi rühren. Die Sauce mit Salz und Pfeffer abschmecken, bei Bedarf mit hellem Saucenbinder binden.Die Sauce mit den Nudeln anrichten.Dazu schmeckt ein grüner Salat.'

We first consider all the words in the Ingredients column to create a named entity model for our ingredients. 
spaCy https://spacy.io/ an open-source software library for advanced natural language processing is used for the purpose. 

``` !python -m spacy download de_core_news_sm

import spacy
nlp = spacy.load('de_core_news_sm', disable=['parser', 'tagger', 'ner'])
```

We first clean the ingredients column of any stop words and then tokenise each unique word found in the ingredients column to form our required named-entities. 
``` set([t.text for t, l in zip(tokenized[4], labels[4]) if l])``` 

For example, from recipe number 4, we find these tokens for our named entity model. 
{'Frischkäse',
 'Kochschinken',
 'Kohlrabi',
 'Milch',
 'Petersilie',
 'Pfeffer',
 'Salz',
 'Saucenbinder'}
 
 

