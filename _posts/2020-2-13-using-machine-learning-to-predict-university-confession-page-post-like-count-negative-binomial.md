---

firstPublishedAt: 1581616038226
latestPublishedAt: 1581616038226
slug: using-machine-learning-to-predict-university-confession-page-post-like-count-negative-binomial
title: "Using Machine Learning to predict University Confession Page post like count (Negative Binomial Model)"

---

![Exefess](https://cdn-images-1.medium.com/max/5084/1*Y8GjMawVlYcoay66vE5YFw.png)

University confession pages have exploded in popularity in recent years. 19,526 people are following the Durham University page “[Durfess](http://facebook.com/Durfess)”, that is 104% of Durham’s student population [(2018/19)](https://www.dur.ac.uk/about/facts/).

Exeter University had the original “ExeHonestly” page shut down, and clones appeared and gained popularity within hours.

These pages are useful not just for students, who can post their frustrations, crushes and weird ideas. These pages could be a goldmine for advertisers. If their brand is named in a popular post on one of these pages, nearly the whole student facebook population are likely to see it (Pages like this grow through friends tagging each other). Best of all — due to the anonymity afforded to everyone who posts to the page, it could be completely free for a marketing team to make a post innocuously naming their brand and propelling it into a student populations consciousness. Therefore, it would make sense that they’d want to know how to make a successful post.

In this article, I will be using “Likes” as a measurement of success. There is probably a better way to combine all reactions, comments and shares to measure success of a post, but that is beyond the scope of this.

---

# **Collecting the data**

To aid our analysis through broadening our dataset, I am going to analyse both “Exefess” and “Durfess” together. Anecdotally, the demographics (other than geography) of these pages are very similar, and the differences will be accounted for during regression through having an indicator variable.

I am using a python module named “[Facebook-Scraper](https://pypi.org/project/facebook-scraper/)” to scrape posts from these pages.

First, I am going to scrape 50,000 pages of each (should be enough to capture all), and mark each with a 1, depending on which page they are from.

```
df_durfess = pd.DataFrame(get_posts("Durfess", pages=50000))
df_durfess['durfess'] = 1

df_exefess = pd.DataFrame(get_posts("ExeFess-110674570384545", pages=50000))
df_exefess['exefess'] = 1
```

---

# Cleaning Data

Merging the datasets, removing the new lines and removing the common #Durfess and #Exefess posts. Other punctuation will be removed later.

```
df = pd.merge(df_durfess, df_exefess, how='outer')
df = df.replace({r'\s+$': '', r'^\s+': ''},
                regex=True).replace(r'\n',  ' ', regex=True)
df.replace(r'\\s', '', regex=True, inplace=True)
df.replace(r'#.+?\b', "", regex=True, inplace=True)
```

We are going to remove any shared posts. This is because on these confession pages, these are adverts for t-shirts or the sharing of the submission link. These are not relevant to our analysis and so it makes sense to remove all of them.

```
df['shared_text'].replace('', np.nan, inplace=True)
df.fillna({'durfess': 0, 'exefess': 0, 'shared_text': 0}, inplace=True)
df = df[df.shared_text == 0]
```

Next, I am going to add indicator variables for whether the post contains a link or an image. We will then remove the image’s and links. A future version of this analysis could analyse the contents.

```
df['haslink'] = df['link'].apply(lambda x: 0 if x == None else 1)
df['haspicture'] = df['image'].apply(lambda x: 0 if x == None else 1)
```

Dropping unneeded columns.

```
df.drop(columns=[“link”, “image”, “shared_text”,
 “text”, “shares”, “post_url”], inplace=True)
df
```

![Our Dataset after cleaning.](https://cdn-images-1.medium.com/max/3518/1*MjMFSE165UXa6_DWo315vw.png)

---

# Data Exploration

![](https://cdn-images-1.medium.com/max/2322/1*8gKZ_laNQOV4o20r8uM6Bg.png)

Mean likes is about 58, but max is 18,405. This, and the 75th percentile both suggest some extreme values. (Comments is even more extreme, so we’ll have to bare that in mind).

**Boxplots**

```
fig_bp, (ax1_bp, ax2_bp) = plt.subplots(1, 2, sharey=True, figsize=(15,8))

durfessdata = df[df['durfess'] == 1]
exefessdata = df[df['exefess'] == 1]

ax1_bp.boxplot(durfessdata['likes'])
ax2_bp.boxplot(exefessdata['likes'])
ax1_bp.set_title('Durfess')
ax2_bp.set_title('Exefess')

plt.show()
```

![](https://cdn-images-1.medium.com/max/2002/1*I6MLZNFACYr7TCVSXHxOzA.png)

As you can see, the outlier is a big one, and comes from the Durfess data. Let’s look at these individual observations and see if there is anything special about them.

```
df[df['likes'] > 2500]
```

![](https://cdn-images-1.medium.com/max/3038/1*jjP9JS8RW1gIJ2ALbUxO_A.png)

![The Durfess post with 18,000 likes.](https://cdn-images-1.medium.com/max/2524/1*YWO0AO3deLn6bfO16y9Dlg.png)

Obviously, the best way to amass a lot of likes is to post a picture of a cat in a magical hat during exam season. I am going to class this as an outlier and remove it.

```
df=df[df['likes'] > 3000].copy()
```

---

# Extracting the topic from the text

I am going to use an unsupervised LDA model to cluster the latent topics from the text, and see if there is any benefit in posting about a certain topic to increase the number of likes.

#### Text Cleaning

First, the text needs to be cleaned. There are many articles that go further in to depth with this, so I will gloss over here.

```
from gensim.utils import simple_preprocess
import gensim
import re
data = df['post_text'].values.tolist()
data = [re.sub('#\S+ *', '', sent) for sent in data]
data = [re.sub('\s+', '', sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]

def prepare_data(data):
    for sentence in data:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data = list(prepare_data(data))
```

Send data to list, remove punctuation, and use gensim simple preprocess to process.

```
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 'post', 'page',
                   'use', 'tag', 'durfess', 'submit', 'exefess'])
```

Remove stopwords, these are some very common words that are not relevant to the topic, and would only be there because they are on a university confession page.

```
import spacy
data = [[word for word in simple_preprocess(
    str(doc)) if word not in stop_words] for doc in data]
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# 'NOUN', 'ADJ', 'VERB', 'ADV'
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in [
                         '-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

data_lemmatized = lemmatization(
    data, allowed_postags=["NOUN", "VERB"])  # select noun and verb
df['post_text2'] = data_lemmatized
sentences_ready = []
allowed_postags = ["NOUN", "VERB"]
for sentence in data:
    doc = nlp(" ".join(sentence))
    sentences_ready.append(" ".join([token.lemma_ if token.lemma_ not in [
                           '-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
data_ready = []
for doc in sentences_ready:
    data_ready.append(gensim.utils.simple_preprocess(doc))
dictionary = gensim.corpora.Dictionary(data_ready)
bow_corpus = [dictionary.doc2bow(doc) for doc in data_ready]
```

Lemmatize (Reduced to stem) and convert to bag of words and dictionary.

**Calculate ideal number of topics through coherence.**

```
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(
            corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(
    dictionary=dictionary, corpus=bow_corpus, texts=data_ready, start=5, limit=100, step=2)
```

Test the LDA model for coherence with a different number of topics each time. (This takes a while).

Selecting the best model.

```
limit = 100
start = 5
step = 2
x = range(start, limit, step)
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

fig_ch, ax_ch=plt.subplots()
ax_ch.plot(x, coherence_values)
```

![](https://cdn-images-1.medium.com/max/9600/1*49bT_TIfniGRyuHp8iTgCw.png)

Optimal topics seems to be 7.

```
optimal_model_number = 1
optimal_model = model_list[optimal_model_number]
num_topics = x[optimal_model_number]
```

This next bit is taken from [this blog post](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/). It is a very elegant way of turning this modelling back into a dataframe that we can use for our regression analysis.

```
def format_topics_sentences(ldamodel=optimal_model, corpus=bow_corpus, texts=data_ready):
    # Init output
    sent_topics_df = pd.DataFrame()

# Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series(
                    [int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic',
                              'Perc_Contribution', 'Topic_Keywords']

# Add original text to the end of the output
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(
    ldamodel=optimal_model, corpus=bow_corpus, texts=data_ready)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No',
                             'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']

# Show
df_dominant_topic.head(10)
```

![](https://cdn-images-1.medium.com/max/2730/1*FhGu3iXqRcXCUMFJe9Yang.png)

Merging this dataframe with our original.

```
df.reset_index(inplace=True)
df = pd.merge(df, df_dominant_topic, left_on='index', right_on='Document_No')
df
```

# Final Data Prep

I am going to drop any observations that don’t have any text (after our cleaning), images or links. This is because they will add nothing to our analysis.

```
df['post_text2'].replace('', 0, inplace=True)
df = df.drop(df[(df['post_text2'] == '') & (
    df['haslink']+df['haspicture'] == 0)].index)
```

Giving any data with no text, but link or image their only topic.

```
df['Dominant_Topic'] = df.apply(
    lambda row: num_topics+1 if row['post_text2'] == 0 else row['Dominant_Topic'], axis=1)
```

# Sentiment Analysis

How positive a post is could be related to how many likes it gets. Lets account for this by using textblob to analyse the sentiment of posts.

```
from textblob import TextBlob as tb

def sentiment(text):
    post = tb(text)
    post_sentiment = post.sentiment.polarity
    return post_sentiment

df['post_text'] = df['post_text'].apply(
    lambda text: str(text))  # convert text into string
df['t_sentiment'] = df['post_text'].apply(lambda post: sentiment(post))
```

**Adding Time**

Adding the time will allow us to measure whether the day of the week has any effect.

```
ds = df.time
df['Month'] = ds.dt.month
df['Day_Of_Week'] = ds.dt.dayofweek
df['Day'] = ds.dt.day
```

---

# Implementing a Negative Binomial Regression

**Regression Goal**: Predict the number of likes a post will receive, based on factors included in the post.

I am using a negative binomial regression for 2 reasons. The first is that we have count data. This data is not continuous (Likes are always integers), and therefore that along with the extreme positive skew, rules out a linear regression.

A poisson regression model is usually recommended for data of this type, though a poisson regression has the requirements that **mean=variance. **In our data:

```
print("Skewness: %f" % df['likes'].skew())
print("Kurtosis: %f" % df['likes'].kurt())
print("Mean: ", df['likes'].mean())
print("Variance: ", df['likes'].var())

Skewness: 7.585360
Kurtosis: 124.327838
Mean:  54.166710770044986
Variance:  12317.058489204572
```

Negative Binomial Regression, however, does not require an equal mean and variance, and is used when the conditional variance exceeds the conditional mean. However, to eventually test a negative binomial regression, we have to run a poisson regression first.

```
mask = np.random.rand(len(df)) < 0.9
df_train = df[mask]
df_test = df[~mask]

explanatory_var = """likes ~ comments  + durfess + haslink + haspicture + Dominant_Topic + Month + Day_Of_Week + Day + t_sentiment"""

y_train, X_train = dmatrices(explanatory_var, df_train, return_type='dataframe')
y_test, X_test = dmatrices(explanatory_var, df_test, return_type='dataframe')

poisson_reg=sm.GLM(y_train,X_train,family=sm.families.Poisson()).fit()

print(poisson_reg.summary())
```

![](https://cdn-images-1.medium.com/max/2594/1*Fht56ws2FiaEiytK4DtrfA.png)

This is obviously not a statistically sound regression. The Deviance and chi² are massive, and therefore we almost certainly do not have a well fitted model. The 5% confidence level with 1000+ degrees of freedom is 1074.679, about 21 times smaller than our deviance. Therefore this is not a good representation of the data.

A negative binomial regression, however does not require a variance=mean, and the most common form of Negative Binomial Regression, nb2, makes the assumption that Variance=mean+alpha\*mean²

![Variance=mean+alpha*mean²](https://cdn-images-1.medium.com/max/1552/1*gPYQIL52jPgQK_cmjaQUJw.png)

We want to find what value of alpha is right for our dataset.

**Auxiliary OLS Regression without a constant to find alpha**

We can use the “mu” or “lambda” values gained from the poisson regression, in order to fit an auxiliary OLS regression that fits our data a lot better. This value is a fitted rate for each observation, and we can use it in the formula seen below.

![](https://cdn-images-1.medium.com/max/2648/1*1fm2r1lC0P0pdhAIJkkpSA.png)

The left hand side of the above equation is equivalent to the coefficient on an independent variable in an OLS regression (plus a zero constant), while the right hand side of the regression is equivalent to a dependent variable. We can then use the ordinary least squares method to find alpha.

Lets add our lambda_i values above to the dataframe.

```
df_train['likes_lambda'] = poisson_training_results.mu
```

And then implement the above equation

```
df_train['y'] = df_train.apply(lambda row: ((row['likes'] - row['likes_lambda'])**2 - row['likes']) / row['likes_lambda'], axis=1)
```

And run the OLS regression

```
ols_vars = """y ~ likes_lambda - 1"""
aux_regression= smf.ols(ols_vars, df_train).fit()
```

Our alpha value (the Beta_1 from the auxiliary regression) equals **0.993958.**

The t value for this coefficient is **10.503339. **For our degrees of freedom and our significance level (5%) the t statistic is 1.645302. 10.5 is obviously higher than this, and therefore we can reject the null hypothesis that alpha=0. This means that our NB2 variance calculation is better than the poisson assumption variance=mean (because alpha is statistically significant).

**Using this Alpha value in our NB2 model.**

We can then use this value of alpha, to run a negative binomial model with our value of alpha calculated from the auxiliary regression.

```
alpha=aux_regression.params[0]
nb2_regression = sm.GLM(y_train,X_train,family=sm.families.NegativeBinomial(alpha=alpha)).fit()
print(nb2_regression.summary())
```

![](https://cdn-images-1.medium.com/max/2614/1*DEjeEqkAbuglLAAVdYoDZQ.png)

Firstly, looking at the statistical significance. At the 5% significance level, “haslink” (whether the post has a link or not), Dominant Topic, Day of the week and t_sentiment are not statistically significant from 0.

This may suggest that the content of the post (Whether it has a link, what topic it is related to, and how positive/negative is) have no effect on the number of likes.

Starting with dominant topic, I find this extremely hard to believe. I would think that that would be the driving force. This is probably down to our methods of analysing the data. If we went through manually and marked the posts topic, this would probably have more of a bearing. For now, we have used unsupervised methods that probably do not fit the dataset very well. This means that the topics are so broad that they have no impact on the number of likes a post gets.

Similarly, the analysis of the sentiment is based on statistically inefficient, unsupervised methods. However, the sentiment of a post is probably not a good indicator of the number of likes. Anecdotally, negative posts and positive posts receive about equal numbers of likes.

Whether we have a link or not is probably not relevant because of the small number of posts that actually do, compared to the size of the dataset.

```
df["haslink"].value_counts()

0    3756
1      23
```

---

Lets remove the statistically insignificant variables (I am going to keep t_sentiment as it is significant at the 10% level, at that is probably within the limits of tenousity for this analysis)

With the statistically insignificant variables removed:

![](https://cdn-images-1.medium.com/max/2556/1*y0dKNtV1t-kNkpBMrmQdmQ.png)

Making predictions using our test data

```
nb2_test_predictions= nb2_regression.get_prediction(X_test)
```

Plotting predicted vs actual counts

```
predictions = nb2_test_predictions.summary_frame()
predicted_counts=predictions['mean']
actual_counts = y_test['likes']

fig, ax = plt.subplots(figsize=(20,15))
predicted, = plt.plot(X_test.index, predicted_counts, label='Predicted counts',color="#5B85AA")
actual, = plt.plot(X_test.index, actual_counts, label='Actual counts', color="#F46036")
plt.legend(handles=[predicted, actual])
ax.set_title("Likes: Predicted vs Actual Counts")
ax.set_ylim(0,900)
fig.tight_layout()
plt.show()
```

![](https://cdn-images-1.medium.com/max/2864/1*PiHeml05z0N2sDjyH6twMw.png)

Our model does alright. The actual counts are a lot more volatile, but that is to be expected. Our model follows the trend pretty well. Let’s go back to our results and analyse them in a bit more detail.

---

#### **Statistically significant coefficients.**

To work out the IRR (Incidence Rate Ratio), and therefore the expected increase or decrease as we increase independent variables, we use the formula

![](https://cdn-images-1.medium.com/max/740/1*ADKfVIG8h4Nrzv66JRtvFw.png)

Implementing that formula:

```
Intercept      9.842956
comments       1.003344
durfess        5.888823
haspicture     2.819429
month          0.950591
day            0.992481
t_sentiment    1.120381
dtype: float64
```

**Comments**

A comment on a post is predicted to increase the number of likes by about 1, holding all else equal.

**Day and Month**

Day and Month both have negative (statistically significant) coefficients. This suggests that both the later in the year, and the later in the month that a post is posted, the fewer likes it will get.

A reason for this could be that the Durfess page often gets more likes than Exefess, and Exefess posts are only from about september onwards. (This should be accounted for in our “Durfess” variable though).

It may also be because of the skew over summer when students are not at university. There are 6 months in the year before summer, while only 3 after.

**Durfess**

The post being on a Durfess page instead of Exefess, is expected to increase the number of likes by 5.89, holding all else constant. Pretty self explanatory, if you want more likes, post to Durfess, probably because the page itself has more likes.

**Has Picture**

Having a picture is expected to increase the number of likes by 2.82, holding all else constant. I suspect this would be higher had we left the mammoth 30,000 reaction post in the data.

# Overall Takeaways

This analysis could definitely be improved by better topic recognition, sentiment analysis and probably more data. Overall, if you want a lot of likes on your post. Post to Durfess, with a picture, and make sure people comment on it. Make sure you post early in the week and early in the year, but the day does not matter.

This analysis was not as successful as I had hoped. The topic modelling is probably the most interesting thing but turned out to be very statistically insignificant. I am going to try and engineer a better way to model the topic and come back to this analysis.

Thanks for reading!
