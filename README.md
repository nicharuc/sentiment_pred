# sentiment_pred_scopeai

To run application, simply run:
```
python sentiment_pred.py
```

The jupyter notebook, sentiment_pred.ipynb also contains the same code, with some additional hyperparamter tuning steps and keyword analysis.

Steps taken:
1. We read in the file via read_file function that automatically generates tokenized list of reviews
2. We split both positive and negative sets into training (80%) and test set (20%).
3. To train the Word2Vec embeddings, we combine both positive, negative, and unsupervised list of tokenized reviews into gensim's Word2Vec model. We run 10 iterations, with each iterations' learning rate decreasing to help algorithm better converge. We ignore words that have frequency less than 5 (via min_count) because these are not substantial.
4. We check whether the embeddings make sense:
```
w2v.most_similar('typical')

[('standard', 0.6307519674301147),
 ('generic', 0.5929383039474487),
 ('usual', 0.5681242346763611),
 ('ish', 0.5674628019332886),
 ('stereotypical', 0.5471631288528442),
 ('type', 0.5376464128494263),
 ('typically', 0.5306581854820251),
 ('mainstay', 0.527876615524292),
 ('mill', 0.49052008986473083),
 ('oriented', 0.48973286151885986)]
 
 w2v.most_similar('bad')
 
 [('lousy', 0.7388101816177368),
 ('terrible', 0.7335008382797241),
 ('good', 0.7166964411735535),
 ('horrible', 0.713603138923645),
 ('awful', 0.6747781038284302),
 ('crappy', 0.6741146445274353),
 ('poor', 0.6500096321105957),
 ('dumb', 0.6295918226242065),
 ('stupid', 0.6238218545913696),
 ('lame', 0.5957022309303284)]
 ```
 These do make sense, so we proceed to the next steps.
 5. For each review, we iterate through each word, get its embedding from Word2Vec, and get an average of the embeddings of all the words in each individual review. This average will be the X features fed into the model.
 6. We use various models, such as Support Vector Machine, Naive Bayes, Logistics Regression to perform classification and evaluate which model gives the best accuracy on the test set. Some tuning on the learning rate is done on the Logistics Regression model.
 - SVM accuracy: 0.850
 - Naive Bayes accuracy: 0.725
 - Logistics Regression Accuracy: 0.852
 
 7. Keywords Analysis
 We use fitted Logistics Regression (since this is the best model in terms of accuracy) to predict each word's embeddings to get the likelihood of the word pointing to positive sentiment or negative sentiment.
 
 For words that are predicted to be 100% probability of being positive, we weigh the size of the term in the word cloud by frequency of its occurence across reviews as frequency is an indicator of importance:
 
 ![alt text](https://github.com/nicharuc/sentiment_pred_scopeai/blob/master/pos_img.png)
 
 We can see that keywords with embeddings that point out to positive sentiment tend to be those that show the emotions that the movie evoked or the art of the film (e.g. combines, blend, maintains, beautifully, captures..etc).
 
 For words that are close to 0% probability of being positive, we perform the same weighing and generate the following word cloud:
 
![alt text](https://github.com/nicharuc/sentiment_pred_scopeai/blob/master/neg_img.png)

We can see that keywords with embeddings that point out to negative sentiment are those that tend to show how the film failed to evoke a clear interest or profound emotions.
