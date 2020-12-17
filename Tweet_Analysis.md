### EDA, Preprocessing, and Tweet Analysis Notebook


```python
import numpy as np
import pandas as pd
import spacy
import re
import nltk
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import seaborn as sns

from nltk.stem.wordnet import WordNetLemmatizer
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from applesauce import model_scoring, cost_benefit_analysis, evaluate_model
from applesauce import model_opt, single_model_opt

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\josep\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\josep\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\josep\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    


```python
nlp = spacy.load('en_core_web_sm')
```


```python
print(stopwords)
print(nlp.Defaults.stop_words)
```

    <WordListCorpusReader in '.../corpora/stopwords' (not loaded yet)>
    {'seems', 'which', 'thereupon', 'really', 'except', 'ourselves', 'third', 'those', 'never', 'more', 'whither', 'used', 'this', 'two', '’ve', 'whom', 'nothing', 'my', '’m', 'at', 'several', 'eleven', 'part', 'seem', 'when', 'whereupon', 'nor', 'beforehand', 'wherever', 'quite', 'thus', 'could', '’ll', 'again', 'just', 'these', 'see', 'via', 'being', 'over', 'less', 'perhaps', 'next', 'enough', 'into', 'and', 'around', 'rather', 'n‘t', 'unless', 'moreover', 'else', 'sometimes', 'his', 'above', 'thereby', 'most', 'however', 'where', 'amongst', 'beyond', 'through', 'empty', 'somewhere', 'became', 'other', 'he', 'same', 'yet', 'had', 'become', 'afterwards', 'though', 'five', 'that', 'was', 'call', 'each', 'yourselves', 'all', 'then', 'whence', 'side', 'go', 'fifty', 'twenty', 'very', 'per', 'between', 'someone', 'still', 'six', 'neither', 'keep', 'indeed', 'than', 'twelve', '‘m', 'against', 'doing', 'yours', 'everywhere', 'whereas', 'down', 'further', 'whole', 'of', 'whenever', 'top', 'anyone', 'to', 'together', 'below', 'out', 'within', 'here', 'whose', 'hundred', 'latterly', 'fifteen', 'many', 'mostly', 'who', 'serious', 'ca', 'now', 'beside', 'none', 'because', 'in', 'bottom', 'ever', 'every', 'might', 'former', 'various', 'since', 'either', '‘ve', 'once', 'she', 'elsewhere', 'whoever', 'hereby', 'you', 'whatever', 'hereupon', 'has', 'have', 'four', 'ours', 'say', 'will', 'seemed', 'anyhow', 'sixty', 'everyone', 'so', 'upon', 'why', 'even', 'amount', 'cannot', 'or', 'although', 'there', 'itself', 'give', 'can', 'make', 'front', 'were', 'sometime', 'always', 'are', 'may', 'name', '‘d', 'what', 'namely', 'eight', 'after', 'along', 'anywhere', 'made', 'no', 'latter', 'something', 'others', 'forty', 'by', 'some', 'three', 'becomes', 'becoming', '’s', 'herein', 'during', 'but', 'wherein', 'without', 'is', 'full', 'its', 'almost', 'anything', '‘s', 'from', 'using', 'done', 'an', "'ve", 'both', 'off', 'therein', 'onto', 'somehow', 'for', 'the', 'our', "'re", 'own', 'him', "'ll", 'does', 'been', 'besides', 'whether', 'her', 'before', 'last', 'toward', 'show', 'herself', 'therefore', 'their', 'move', 'nine', 'anyway', 'hence', 'alone', 'myself', 'how', "n't", '’d', 'throughout', 'them', 'a', 'ten', 'formerly', 'me', 'thru', 'himself', 'must', 'up', "'s", 'about', 'on', 'please', 'across', 'towards', 'get', 'if', 'as', 'regarding', 'any', 'would', 'should', 'too', 're', 'we', 'nobody', 'under', 'noone', 'yourself', 'with', 'often', 'nowhere', 'do', 'while', 'among', 'otherwise', 'am', 'well', 'back', 'thence', 'hereafter', 'take', 'not', 'put', 'least', 'everything', 'they', 'i', 'did', 'it', "'d", 'due', 'whereby', 'another', 'already', 'us', 'thereafter', 'until', '’re', 'only', 'few', 'themselves', 'hers', 'n’t', '‘ll', 'your', 'also', 'such', 'nevertheless', 'much', 'be', 'seeming', 'meanwhile', 'mine', '‘re', 'one', 'whereafter', "'m", 'behind', 'first'}
    


```python
df = pd.read_csv('data/product_tweets.csv',encoding='latin1')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>emotion_in_tweet_is_directed_at</th>
      <th>is_there_an_emotion_directed_at_a_brand_or_product</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['is_there_an_emotion_directed_at_a_brand_or_product'].unique()
```




    array(['Negative emotion', 'Positive emotion',
           'No emotion toward brand or product', "I can't tell"], dtype=object)




```python
df = df.rename(columns= {'is_there_an_emotion_directed_at_a_brand_or_product'
                         :'Emotion','emotion_in_tweet_is_directed_at': 'Platform'})
```


```python
df = df.rename(columns= {'tweet_text': 'Tweet'})
```


```python
df.head() # want to remove the @'name' in the tweet 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Platform</th>
      <th>Emotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_dummify = pd.get_dummies(df['Emotion'])
```


```python
df_dummify.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>I can't tell</th>
      <th>Negative emotion</th>
      <th>No emotion toward brand or product</th>
      <th>Positive emotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_dummify.sum() # class bias 
```




    I can't tell                           156
    Negative emotion                       570
    No emotion toward brand or product    5389
    Positive emotion                      2978
    dtype: int64




```python
df.info()
df = pd.merge(df, df_dummify, how='outer',on=df.index) # ran this code, dummify emotion data
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9093 entries, 0 to 9092
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   Tweet     9092 non-null   object
     1   Platform  3291 non-null   object
     2   Emotion   9093 non-null   object
    dtypes: object(3)
    memory usage: 213.2+ KB
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 9093 entries, 0 to 9092
    Data columns (total 8 columns):
     #   Column                              Non-Null Count  Dtype 
    ---  ------                              --------------  ----- 
     0   key_0                               9093 non-null   int64 
     1   Tweet                               9092 non-null   object
     2   Platform                            3291 non-null   object
     3   Emotion                             9093 non-null   object
     4   I can't tell                        9093 non-null   uint8 
     5   Negative emotion                    9093 non-null   uint8 
     6   No emotion toward brand or product  9093 non-null   uint8 
     7   Positive emotion                    9093 non-null   uint8 
    dtypes: int64(1), object(3), uint8(4)
    memory usage: 390.7+ KB
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key_0</th>
      <th>Tweet</th>
      <th>Platform</th>
      <th>Emotion</th>
      <th>I can't tell</th>
      <th>Negative emotion</th>
      <th>No emotion toward brand or product</th>
      <th>Positive emotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.rename(columns = {"I can't tell": "Uncertain", 'Negative emotion': 'Negative'
                          , 'No emotion toward brand or product': 'No Emotion'
                          , 'Positive emotion':'Positive'})
```


```python
df = df.drop(columns='key_0')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Platform</th>
      <th>Emotion</th>
      <th>Uncertain</th>
      <th>Negative</th>
      <th>No Emotion</th>
      <th>Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
corpus = list(df['Tweet'])
corpus[:10]
```




    ['.@wesley83 I have a 3G iPhone. After 3 hrs tweeting at #RISE_Austin, it was dead!  I need to upgrade. Plugin stations at #SXSW.',
     "@jessedee Know about @fludapp ? Awesome iPad/iPhone app that you'll likely appreciate for its design. Also, they're giving free Ts at #SXSW",
     '@swonderlin Can not wait for #iPad 2 also. They should sale them down at #SXSW.',
     "@sxsw I hope this year's festival isn't as crashy as this year's iPhone app. #sxsw",
     "@sxtxstate great stuff on Fri #SXSW: Marissa Mayer (Google), Tim O'Reilly (tech books/conferences) &amp; Matt Mullenweg (Wordpress)",
     '@teachntech00 New iPad Apps For #SpeechTherapy And Communication Are Showcased At The #SXSW Conference http://ht.ly/49n4M #iear #edchat #asd',
     nan,
     '#SXSW is just starting, #CTIA is around the corner and #googleio is only a hop skip and a jump from there, good time to be an #android fan',
     'Beautifully smart and simple idea RT @madebymany @thenextweb wrote about our #hollergram iPad app for #sxsw! http://bit.ly/ieaVOB',
     'Counting down the days to #sxsw plus strong Canadian dollar means stock up on Apple gear']



### Tokenize the Words


```python
tokenz = word_tokenize(','.join(str(v) for v in corpus))
```


```python
tokenz[:10]
```




    ['.', '@', 'wesley83', 'I', 'have', 'a', '3G', 'iPhone', '.', 'After']



### Create Stopwords List


```python
stopword_list = list(nlp.Defaults.stop_words)
len(nlp.Defaults.stop_words)
```




    326




```python
stopword_list
```




    ['seems',
     'which',
     'thereupon',
     'really',
     'except',
     'ourselves',
     'third',
     'those',
     'never',
     'more',
     'whither',
     'used',
     'this',
     'two',
     '’ve',
     'whom',
     'nothing',
     'my',
     '’m',
     'at',
     'several',
     'eleven',
     'part',
     'seem',
     'when',
     'whereupon',
     'nor',
     'beforehand',
     'wherever',
     'quite',
     'thus',
     'could',
     '’ll',
     'again',
     'just',
     'these',
     'see',
     'via',
     'being',
     'over',
     'less',
     'perhaps',
     'next',
     'enough',
     'into',
     'and',
     'around',
     'rather',
     'n‘t',
     'unless',
     'moreover',
     'else',
     'sometimes',
     'his',
     'above',
     'thereby',
     'most',
     'however',
     'where',
     'amongst',
     'beyond',
     'through',
     'empty',
     'somewhere',
     'became',
     'other',
     'he',
     'same',
     'yet',
     'had',
     'become',
     'afterwards',
     'though',
     'five',
     'that',
     'was',
     'call',
     'each',
     'yourselves',
     'all',
     'then',
     'whence',
     'side',
     'go',
     'fifty',
     'twenty',
     'very',
     'per',
     'between',
     'someone',
     'still',
     'six',
     'neither',
     'keep',
     'indeed',
     'than',
     'twelve',
     '‘m',
     'against',
     'doing',
     'yours',
     'everywhere',
     'whereas',
     'down',
     'further',
     'whole',
     'of',
     'whenever',
     'top',
     'anyone',
     'to',
     'together',
     'below',
     'out',
     'within',
     'here',
     'whose',
     'hundred',
     'latterly',
     'fifteen',
     'many',
     'mostly',
     'who',
     'serious',
     'ca',
     'now',
     'beside',
     'none',
     'because',
     'in',
     'bottom',
     'ever',
     'every',
     'might',
     'former',
     'various',
     'since',
     'either',
     '‘ve',
     'once',
     'she',
     'elsewhere',
     'whoever',
     'hereby',
     'you',
     'whatever',
     'hereupon',
     'has',
     'have',
     'four',
     'ours',
     'say',
     'will',
     'seemed',
     'anyhow',
     'sixty',
     'everyone',
     'so',
     'upon',
     'why',
     'even',
     'amount',
     'cannot',
     'or',
     'although',
     'there',
     'itself',
     'give',
     'can',
     'make',
     'front',
     'were',
     'sometime',
     'always',
     'are',
     'may',
     'name',
     '‘d',
     'what',
     'namely',
     'eight',
     'after',
     'along',
     'anywhere',
     'made',
     'no',
     'latter',
     'something',
     'others',
     'forty',
     'by',
     'some',
     'three',
     'becomes',
     'becoming',
     '’s',
     'herein',
     'during',
     'but',
     'wherein',
     'without',
     'is',
     'full',
     'its',
     'almost',
     'anything',
     '‘s',
     'from',
     'using',
     'done',
     'an',
     "'ve",
     'both',
     'off',
     'therein',
     'onto',
     'somehow',
     'for',
     'the',
     'our',
     "'re",
     'own',
     'him',
     "'ll",
     'does',
     'been',
     'besides',
     'whether',
     'her',
     'before',
     'last',
     'toward',
     'show',
     'herself',
     'therefore',
     'their',
     'move',
     'nine',
     'anyway',
     'hence',
     'alone',
     'myself',
     'how',
     "n't",
     '’d',
     'throughout',
     'them',
     'a',
     'ten',
     'formerly',
     'me',
     'thru',
     'himself',
     'must',
     'up',
     "'s",
     'about',
     'on',
     'please',
     'across',
     'towards',
     'get',
     'if',
     'as',
     'regarding',
     'any',
     'would',
     'should',
     'too',
     're',
     'we',
     'nobody',
     'under',
     'noone',
     'yourself',
     'with',
     'often',
     'nowhere',
     'do',
     'while',
     'among',
     'otherwise',
     'am',
     'well',
     'back',
     'thence',
     'hereafter',
     'take',
     'not',
     'put',
     'least',
     'everything',
     'they',
     'i',
     'did',
     'it',
     "'d",
     'due',
     'whereby',
     'another',
     'already',
     'us',
     'thereafter',
     'until',
     '’re',
     'only',
     'few',
     'themselves',
     'hers',
     'n’t',
     '‘ll',
     'your',
     'also',
     'such',
     'nevertheless',
     'much',
     'be',
     'seeming',
     'meanwhile',
     'mine',
     '‘re',
     'one',
     'whereafter',
     "'m",
     'behind',
     'first']




```python
stopword_list.extend(string.punctuation)
```


```python
len(stopword_list)
```




    358




```python
stopword_list.extend(stopwords.words('english'))
```


```python
len(stopword_list)
```




    537




```python
additional_punc = ['“','”','...',"''",'’','``','https','rt','\.+']
stopword_list.extend(additional_punc)
stopword_list[-10:]
```




    ["wouldn't", '“', '”', '...', "''", '’', '``', 'https', 'rt', '\\.+']



### Remove stopwords and additional punctuation from the data


```python
stopped_tokenz = [word.lower() for word in tokenz if word.lower() not in stopword_list]
```


```python
freq = FreqDist(stopped_tokenz)
freq.most_common(50)
```




    [('sxsw', 9418),
     ('mention', 7120),
     ('link', 4313),
     ('google', 2593),
     ('ipad', 2432),
     ('apple', 2301),
     ('quot', 1696),
     ('iphone', 1516),
     ('store', 1472),
     ('2', 1114),
     ('new', 1090),
     ('austin', 959),
     ('amp', 836),
     ('app', 810),
     ('circles', 658),
     ('launch', 653),
     ('social', 647),
     ('android', 574),
     ('today', 574),
     ('network', 465),
     ('ipad2', 457),
     ('pop-up', 420),
     ('line', 405),
     ('free', 387),
     ('called', 361),
     ('party', 346),
     ('sxswi', 340),
     ('mobile', 338),
     ('major', 301),
     ('like', 290),
     ('time', 271),
     ('temporary', 264),
     ('opening', 257),
     ('possibly', 240),
     ('people', 226),
     ('downtown', 225),
     ('apps', 224),
     ('great', 222),
     ('maps', 219),
     ('going', 217),
     ('check', 216),
     ('mayer', 214),
     ('day', 214),
     ('open', 210),
     ('popup', 209),
     ('need', 205),
     ('marissa', 189),
     ('got', 185),
     ('w/', 182),
     ('know', 180)]



### Lemmatize the Data and use Regex to find and remove URL's, Tags, other misc


```python
additional_misc = ['sxsw','mention',r'[a-zA-Z]+\'?s]',r"(http[s]?://\w*\.\w*/+\w+)"
                   ,r'\#\w*',r'RT [@]?\w*:',r'\@\w*',r"\d$",r"^\d"
                   ,r"([a-zA-Z]+(?:'[a-z]+)?)",r'\d.',r'\d','RT',r'^http[s]?','za'] #[A-Z]{2,20} remove caps like MAGA and CDT
stopword_list.extend(additional_misc)
stopword_list.extend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
```


```python
lemmatizer = WordNetLemmatizer()
```


```python
clean_stopped_tokenz = [word.lower() for word in stopped_tokenz if word not in stopword_list]
clean_lemmatized_tokenz = [lemmatizer.lemmatize(word.lower()) for word in stopped_tokenz if word not in stopword_list]
```


```python
freq_clean_lemma = FreqDist(clean_lemmatized_tokenz)
freq_lemma = freq_clean_lemma.most_common(5000)
freq_lemma2 = freq_clean_lemma.most_common(25)
```


```python
total_word_count = len(clean_lemmatized_tokenz)
```


```python
lemma_word_count = sum(freq_clean_lemma.values()) # just a number
```


```python
for word in freq_lemma2:
    normalized_freq = word[1] / lemma_word_count
    print(word, "----", "{:.3f}".format(normalized_freq*100),"%")
```

    ('link', 4324) ---- 5.004 %
    ('google', 2594) ---- 3.002 %
    ('ipad', 2432) ---- 2.814 %
    ('apple', 2304) ---- 2.666 %
    ('quot', 1696) ---- 1.963 %
    ('iphone', 1516) ---- 1.754 %
    ('store', 1511) ---- 1.749 %
    ('new', 1090) ---- 1.261 %
    ('austin', 960) ---- 1.111 %
    ('amp', 836) ---- 0.967 %
    ('app', 810) ---- 0.937 %
    ('launch', 691) ---- 0.800 %
    ('circle', 673) ---- 0.779 %
    ('social', 647) ---- 0.749 %
    ('android', 574) ---- 0.664 %
    ('today', 574) ---- 0.664 %
    ('network', 473) ---- 0.547 %
    ('ipad2', 457) ---- 0.529 %
    ('line', 442) ---- 0.512 %
    ('pop-up', 422) ---- 0.488 %
    ('free', 387) ---- 0.448 %
    ('party', 386) ---- 0.447 %
    ('called', 361) ---- 0.418 %
    ('mobile', 340) ---- 0.393 %
    ('sxswi', 340) ---- 0.393 %
    


```python
# from wordcloud import WordCloud

# ## Initalize a WordCloud with our stopwords_list and no bigrams
# wordcloud = WordCloud(stopwords=stopword_list,collocations=False)

# ## Generate wordcloud from stopped_tokens
# wordcloud.generate(','.join(clean_lemmatized_tokenz))

# ## Plot with matplotlib
# plt.figure(figsize = (12, 12), facecolor = None) 
# plt.imshow(wordcloud) 
# plt.axis('off')
```


```python
bigram_measures = nltk.collocations.BigramAssocMeasures()
tweet_finder = nltk.BigramCollocationFinder.from_words(clean_lemmatized_tokenz)
tweets_scored = tweet_finder.score_ngrams(bigram_measures.raw_freq)
```


```python
word_pairs = pd.DataFrame(tweets_scored, columns=["Word","Freq"]).head(20)
word_pairs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Word</th>
      <th>Freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(apple, store)</td>
      <td>0.006920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(social, network)</td>
      <td>0.005277</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(new, social)</td>
      <td>0.004837</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(google, launch)</td>
      <td>0.003912</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(link, google)</td>
      <td>0.003877</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(network, called)</td>
      <td>0.003784</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(called, circle)</td>
      <td>0.003634</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(today, link)</td>
      <td>0.003437</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(major, new)</td>
      <td>0.003356</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(iphone, app)</td>
      <td>0.003333</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(launch, major)</td>
      <td>0.003264</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(link, apple)</td>
      <td>0.003055</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(pop-up, store)</td>
      <td>0.002870</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(possibly, today)</td>
      <td>0.002731</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(circle, possibly)</td>
      <td>0.002720</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(apple, opening)</td>
      <td>0.002615</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(google, circle)</td>
      <td>0.002430</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(store, austin)</td>
      <td>0.002268</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(temporary, store)</td>
      <td>0.002234</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(downtown, austin)</td>
      <td>0.002199</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig_dims = (20,8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.set(font_scale=2)
sns.set_style("darkgrid")
palette = sns.set_palette("dark")
ax = sns.barplot(x=word_pairs.head(15)['Word'], y=word_pairs.head(15)['Freq'], palette=palette)
ax.set(xlabel="Word Pairs",ylabel="Frequency")
plt.ticklabel_format(style='plain',axis='y')
plt.xticks(rotation=70)
plt.title('Top 15 Word Pairs by Frequency')
plt.show()
```


![png](Tweet_Analysis_files/Tweet_Analysis_44_0.png)



```python
tweet_pmi_finder = nltk.BigramCollocationFinder.from_words(clean_lemmatized_tokenz)
tweet_pmi_finder.apply_freq_filter(5)

tweet_pmi_scored = tweet_pmi_finder.score_ngrams(bigram_measures.pmi)
```


```python
PMI_list = pd.DataFrame(tweet_pmi_scored, columns=["Words","PMI"]).head(20)
PMI_list
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Words</th>
      <th>PMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(û÷sxsw, goûª)</td>
      <td>14.076983</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(jc, penney)</td>
      <td>13.813948</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(knitted, staircase)</td>
      <td>13.813948</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(naomi, campbell)</td>
      <td>13.813948</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(parking, 5-10)</td>
      <td>13.813948</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(pauly, celebs)</td>
      <td>13.813948</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(98, accuracy)</td>
      <td>13.591556</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(aron, pilhofer)</td>
      <td>13.591556</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(charlie, sheen)</td>
      <td>13.591556</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(lynn, teo)</td>
      <td>13.591556</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(sheen, goddess)</td>
      <td>13.591556</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(elusive, 'power)</td>
      <td>13.398911</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(zazzlsxsw, youûªll)</td>
      <td>13.398911</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(cameron, sinclair)</td>
      <td>13.398911</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(sinclair, spearhead)</td>
      <td>13.398911</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(staircase, attendance)</td>
      <td>13.398911</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(likeability, virgin)</td>
      <td>13.328521</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(14-day, return)</td>
      <td>13.228986</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(launchrock, comp)</td>
      <td>13.228986</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(participating, launchrock)</td>
      <td>13.228986</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig_dims = (20,8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.set(font_scale=2)
sns.set_style("darkgrid")
palette = sns.set_palette("dark")
ax = sns.barplot(x=PMI_list.head(15)['Words'], y=PMI_list.head(15)['PMI'], palette=palette)
ax.set(xlabel="PMI Pairs",ylabel="Frequency")
plt.ylim([13,14.5])
plt.ticklabel_format(style='plain',axis='y')
plt.xticks(rotation=70)
plt.title('Top 15 Word Pairs by PMI')
plt.show()
```


![png](Tweet_Analysis_files/Tweet_Analysis_47_0.png)



```python
df1 = df
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Platform</th>
      <th>Emotion</th>
      <th>Uncertain</th>
      <th>Negative</th>
      <th>No Emotion</th>
      <th>Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = df1.drop(columns=['Uncertain','No Emotion'])
# Turn negative and positive columns into one column of just negatives and positive.
df1 = df1[df1['Emotion'] != "No emotion toward brand or product"]
df1 = df1[df1['Emotion'] != "I can't tell"]
df1 = df1.drop(columns='Negative')
df1 = df1.rename(columns={'Positive': 'Positive_Bin'})
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Platform</th>
      <th>Emotion</th>
      <th>Positive_Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Create upsampled data, train and test sets



```python
from sklearn.utils import resample
```


```python
df_majority = df1.loc[df1['Positive_Bin']==1]
df_minority = df1.loc[df1['Positive_Bin']==0]
```


```python
df_minority.shape
```




    (570, 4)




```python
df_majority.shape
```




    (2978, 4)




```python
df_min_sample = resample(df_minority, replace=True, n_samples=1000, random_state=42)
```


```python
df_maj_sample = resample(df_majority, replace=True, n_samples=2500, random_state=42)
```


```python
df_upsampled = pd.concat([df_min_sample, df_maj_sample], axis=0)
df_upsampled.shape
```




    (3500, 4)




```python
X, y = df_upsampled['Tweet'], df_upsampled['Positive_Bin']
```

### Train/Test Split


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
scaler_object = MaxAbsScaler()
```


```python
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3548 entries, 0 to 9088
    Data columns (total 4 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   Tweet         3548 non-null   object
     1   Platform      3191 non-null   object
     2   Emotion       3548 non-null   object
     3   Positive_Bin  3548 non-null   uint8 
    dtypes: object(3), uint8(1)
    memory usage: 114.3+ KB
    


```python
y_train.value_counts(0)
y_test.value_counts(1)
```

    2020-12-17 12:53:52,995 : INFO : NumExpr defaulting to 8 threads.
    




    1    0.683429
    0    0.316571
    Name: Positive_Bin, dtype: float64



### Vectorize, Lemmatize with Count Vectorizer and Tf Idf


```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

tokenizer = nltk.TweetTokenizer(preserve_case=False)

vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize,
                             stop_words=stopword_list,decode_error='ignore')
```


```python
# for row in X_train:
#     for word in row:
#         lemmatizer.lemmatize(X_train[row][word])
# return X_train[word][row]
```


```python
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)
```

    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    


```python
ran_for = RandomForestClassifier(class_weight='balanced')
model = ran_for.fit(X_train_count, y_train)
```


```python
y_hat_test = model.predict(X_test_count)
```

### Evaluate Models


```python
evaluate_model(y_test, y_hat_test, X_test_count,clf=model) # 1 denotes Positive Tweet
```

                  precision    recall  f1-score   support
    
               0       0.96      0.83      0.89       277
               1       0.93      0.98      0.96       598
    
        accuracy                           0.94       875
       macro avg       0.95      0.91      0.92       875
    weighted avg       0.94      0.94      0.94       875
    
    


![png](Tweet_Analysis_files/Tweet_Analysis_71_1.png)



```python
tf_idf_vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize,
                                    stop_words=stopword_list,decode_error='ignore')
```


```python
X_train_tf_idf = tf_idf_vectorizer.fit_transform(X_train)
X_test_tf_idf = tf_idf_vectorizer.transform(X_test)
print(X_train_tf_idf.shape)
print(y_train.shape)
```

    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    

    (2625, 4295)
    (2625,)
    


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
ran_for = RandomForestClassifier(class_weight='balanced')
model_tf_idf = ran_for.fit(X_train_tf_idf,y_train)
```


```python
y_hat_tf_idf = model_tf_idf.predict(X_test_count)
```


```python
evaluate_model(y_test, y_hat_tf_idf, X_test_tf_idf,clf=model_tf_idf) # slightly better performance
```

                  precision    recall  f1-score   support
    
               0       0.90      0.64      0.75       277
               1       0.85      0.97      0.91       598
    
        accuracy                           0.86       875
       macro avg       0.88      0.80      0.83       875
    weighted avg       0.87      0.86      0.86       875
    
    


![png](Tweet_Analysis_files/Tweet_Analysis_77_1.png)



```python
ran_for = RandomForestClassifier()
ada_clf = AdaBoostClassifier()
gb_clf = GradientBoostingClassifier()

models = [ran_for, ada_clf, gb_clf]

for model in models:
    single_model_opt(ran_for, X_train_count, y_train, X_test_count, y_test)
```

    Accuracy Score:  0.9371428571428572
    Precision Score:  0.9235569422776911
    Recall Score:  0.9899665551839465
    F1 Score:  0.9556093623890234
    RandomForestClassifier()   0.9371428571428572
    


![png](Tweet_Analysis_files/Tweet_Analysis_78_1.png)


    Accuracy Score:  0.9314285714285714
    Precision Score:  0.9177018633540373
    Recall Score:  0.9882943143812709
    F1 Score:  0.9516908212560387
    RandomForestClassifier()   0.9314285714285714
    


![png](Tweet_Analysis_files/Tweet_Analysis_78_3.png)


    Accuracy Score:  0.9314285714285714
    Precision Score:  0.9164086687306502
    Recall Score:  0.9899665551839465
    F1 Score:  0.9517684887459806
    RandomForestClassifier()   0.9314285714285714
    


![png](Tweet_Analysis_files/Tweet_Analysis_78_5.png)



```python
for model in models:
    single_model_opt(ran_for, X_train_tf_idf, y_train, X_test_tf_idf, y_test)
```

    Accuracy Score:  0.9291428571428572
    Precision Score:  0.9161490683229814
    Recall Score:  0.9866220735785953
    F1 Score:  0.9500805152979066
    RandomForestClassifier()   0.9291428571428572
    


![png](Tweet_Analysis_files/Tweet_Analysis_79_1.png)


    Accuracy Score:  0.9314285714285714
    Precision Score:  0.9190031152647975
    Recall Score:  0.9866220735785953
    F1 Score:  0.9516129032258065
    RandomForestClassifier()   0.9314285714285714
    


![png](Tweet_Analysis_files/Tweet_Analysis_79_3.png)


    Accuracy Score:  0.9314285714285714
    Precision Score:  0.9190031152647975
    Recall Score:  0.9866220735785953
    F1 Score:  0.9516129032258065
    RandomForestClassifier()   0.9314285714285714
    


![png](Tweet_Analysis_files/Tweet_Analysis_79_5.png)



```python
tf_idf_vectorizer.get_feature_names()
```




    ['##sxsw',
     '#10',
     '#106',
     '#11ntc',
     '#1406-08',
     '#15slides',
     '#310409h2011',
     '#4sq',
     '#911tweets',
     '#abacus',
     '#accesssxsw',
     '#accordion',
     '#aclu',
     '#adam',
     '#addictedtotheinterwebs',
     '#adpeopleproblems',
     '#agchat',
     '#agileagency',
     '#agnerd',
     '#allhat3',
     '#alwayshavingtoplugin',
     '#amateurhour',
     '#android',
     "#android's",
     '#androidsxsw',
     '#angrybirds',
     '#annoying',
     '#app',
     '#appcircus',
     '#apple',
     "#apple's",
     '#apple_store',
     '#appleatxdt',
     '#applefanatic',
     '#appletakingoverworld',
     '#apps',
     '#appstore',
     '#art',
     '#assistivetech',
     '#at',
     '#atl',
     '#att',
     '#atx',
     '#atzip',
     '#augcomm',
     '#aus',
     '#austin',
     '#austincrowd',
     '#austinwins',
     '#ausxsw',
     '#bankinnovate',
     '#bankinnovation',
     '#barrydiller',
     '#batterykiller',
     '#battlela',
     '#bavcid',
     '#bawling',
     '#bbq',
     '#behance',
     '#bestappever',
     '#betainvites',
     '#bettercloud',
     '#bettersearch',
     '#betterthingstodo',
     '#beyondwc',
     '#bing',
     '#bizzy',
     '#blackberry',
     '#boom',
     '#booyah',
     '#brainwashed',
     '#brian_lam',
     '#brk',
     '#broadcastr',
     '#browserwars',
     '#cartoon',
     '#catphysics',
     '#cbatsxsw',
     '#ces',
     '#channels',
     '#chargin2diffphonesatonce',
     '#checkins',
     '#circles',
     '#circusmash',
     '#classical',
     '#cloud',
     '#cnet',
     '#cnn',
     '#cnngrill',
     '#comcom',
     '#comments',
     '#confusion',
     '#conversation',
     '#coronasdk',
     '#courtyard',
     '#crazyco',
     '#crowded',
     '#crushit',
     '#csr',
     '#cstejas',
     '#ctia',
     '#curatedebate',
     '#cwc2011',
     '#dairy',
     '#dfcbto',
     '#digitalluxury',
     '#diller',
     '#discovr',
     '#doesdroid',
     '#dokobots',
     '#domo',
     '#dorkinout',
     '#dotco',
     '#duh',
     '#dyac',
     '#earthhour',
     '#ecademy',
     '#edchat',
     '#edtech',
     '#efficient',
     '#emc',
     '#empowered',
     '#enchantment',
     '#entry',
     '#evaporation',
     '#events',
     '#eventseekr',
     '#evolvingworkplace',
     '#fab5',
     '#fail',
     '#fanboy',
     '#fandango',
     '#fastcompanygrill',
     '#fastsociety',
     '#fb',
     '#ff',
     '#fh',
     '#filmaster',
     '#flip-board',
     '#flipboard',
     '#fml',
     '#fmsignal',
     '#foursquare',
     '#friends',
     '#frostwire',
     '#fuckit',
     '#futurecast',
     '#futuremf',
     '#futureoftouch',
     '#fxsw',
     '#gadget',
     '#gadgets',
     '#gamesfortv',
     '#gamestorming',
     '#geek',
     '#geekery',
     '#geekout',
     '#genius',
     '#geogames',
     '#getjarsxsw',
     '#girlcrush',
     '#gitchococktailon',
     '#gonnagetanipad2',
     '#goodcustomerservice',
     '#google',
     '#googlebread',
     '#googlecircles',
     '#googledoodle',
     '#googledoodles',
     '#googleio',
     '#googlemaps',
     '#googleplaces',
     '#gowalla',
     '#gps',
     '#greatergood',
     '#groupchatapps',
     '#groupme',
     '#grrr',
     '#gsdm',
     '#gswsxsw',
     '#guykawasaki',
     '#h4ckers',
     '#hacknews',
     '#happydance',
     '#hashable',
     '#hcsm',
     '#help',
     '#hhrs',
     '#hipstamatic',
     '#hipster',
     '#hireme',
     '#hisxsw',
     '#hollergram',
     '#hollrback',
     '#holytrafficjams',
     '#house',
     '#html5',
     '#idontbelieve',
     '#ie9',
     '#igottagetit',
     '#il',
     '#illmakeitwork',
     '#imanoutcast',
     '#in',
     '#innotribe',
     '#ios',
     '#ipad',
     '#ipad2',
     "#ipad2's",
     '#ipad2time',
     '#ipad_2',
     '#ipaddesignheadaches',
     '#ipadmadness',
     '#iphone',
     '#iphone4',
     '#iphone5',
     '#ipod',
     '#iqlab',
     '#itunes',
     '#iusxsw',
     '#iwantacameraonmyipad',
     '#japan',
     '#jealous',
     '#jk',
     '#jpmobilesummit',
     '#justinjustinjustin',
     '#justmet',
     '#justsayin',
     '#justsaying',
     '#kawasaki',
     '#kids',
     '#killcommunity',
     '#kirkus',
     '#lbs',
     '#leanstartup',
     '#letushopenot',
     '#libraries',
     '#lines',
     '#livingthedream',
     '#lmndst',
     '#logo',
     '#lonely-planet',
     '#lonelyplanet',
     '#longlinesbadux',
     '#looseorganizations',
     '#loveher',
     '#lp',
     '#lxh',
     '#mac',
     '#macallan',
     '#madebymany',
     '#maps',
     '#marissagoogle',
     '#marissamayer',
     '#marissameyer',
     '#marketing',
     '#mashable',
     '#mccannsxsw',
     '#media',
     '#mhealth',
     '#miamibeach',
     '#microformats',
     '#midem',
     '#mindjet',
     '#minimalistprogramming',
     '#mitharvard',
     '#mobile',
     '#mobilefarm',
     '#mophie',
     '#moreknowledge',
     '#musedchat',
     '#music',
     '#musicviz',
     '#mxm',
     '#myegc',
     '#mylunch',
     '#nerd',
     '#nerdcore',
     '#nerdheaven',
     '#nerds',
     '#netflix',
     '#netflixiphone',
     '#networking',
     '#new',
     '#news',
     '#newsapp',
     '#newsapps',
     '#newtwitter',
     '#nfusion',
     '#nokiaconnects',
     '#notionink',
     '#notpouting',
     '#novideo',
     '#nowhammies',
     '#nten',
     '#nudgenudge',
     '#ogilvynotes',
     '#oldschool',
     '#omfg',
     '#omg',
     '#osmpw',
     '#ouch',
     '#owllove',
     '#pakistan',
     '#pandora',
     '#papasangre',
     '#pc',
     '#pgi',
     '#photo',
     '#photos',
     '#photosharing',
     '#pissedimnotgoingtosxsw',
     '#playhopskoch',
     '#playsxsw',
     '#please',
     '#pnid',
     '#poetry',
     '#ponies',
     '#popplet',
     '#poppop',
     '#popupstore',
     '#posterous',
     '#postpc',
     '#poursite',
     '#powermat',
     '#powermatteam',
     '#powermattteam',
     '#precommerce',
     '#prodmktg',
     '#progressbar',
     '#project314',
     '#protip',
     '#psych',
     '#pubcamp',
     '#pushsnowboarding',
     '#qagb',
     '#qrcode',
     '#quibidswin',
     '#rad',
     '#random',
     '#realtalk',
     '#rejection',
     '#retail',
     '#rise_austin',
     '#rji',
     '#saatchiny',
     '#savebrands',
     '#saveustechies',
     '#saysshewithoutanipad',
     '#scoremore',
     '#seattle',
     '#seenocreepy',
     '#sem',
     '#seo',
     '#shame',
     '#shareable',
     '#shocked',
     '#showusyouricrazy',
     '#sightings',
     '#silly',
     '#singularity',
     '#smartcover',
     '#smartphones',
     '#smcomedyfyeah',
     '#smileyparty',
     '#smm',
     '#smtravel',
     '#socbiz',
     '#social',
     '#socialfuel',
     '#socialmedia',
     '#socialmuse',
     '#socialviewing',
     '#socmedia',
     '#sony',
     '#soundcloud',
     '#spiltbeer',
     '#startupbus',
     '#stillonamacbook',
     '#store',
     '#stumbledupon',
     '#suxsw',
     '#swsurrogates',
     '#sxflip',
     '#sxprotect',
     '#sxsh',
     '#sxsw',
     '#sxsw-bound',
     '#sxsw-ers',
     '#sxsw-sters',
     '#sxsw11',
     '#sxsw2011',
     '#sxswbarcrawl',
     '#sxswbuffalo',
     '#sxswchi',
     '#sxsweisner',
     '#sxswgo',
     '#sxswh',
     '#sxswi',
     '#sxswmobileapps',
     '#sxswmoot',
     '#sxswmusic',
     '#sxswmymistake',
     '#sxswnui',
     '#sxswtoolkit',
     '#sxtxstate',
     '#sxwsi',
     '#tablet',
     '#taccsxsw',
     '#tapworthy',
     '#tbwasxsw',
     '#tc',
     '#team_android',
     '#teamandroid',
     '#teamandroidsxsw',
     '#tech',
     '#tech_news',
     '#techenvy',
     '#techiesunite',
     '#technews',
     '#technology',
     '#texasevery',
     '#texting',
     '#thanks',
     '#thankyouecon',
     '#the_daily',
     '#theindustryparty',
     '#theplatform',
     '#thingsthatdontgotogether',
     '#thinmints',
     '#thisisdare',
     '#tigerblood',
     '#tmobile',
     '#tmsxsw',
     '#tnw',
     '#tonchidot',
     '#toodamnlucky',
     '#topnews',
     '#touchingstories',
     '#tradeshow',
     '#travel',
     '#trending',
     '#tsunami',
     '#tt',
     '#tveverywhere',
     '#tweethouse',
     '#tweetignite',
     '#twitter',
     '#tye',
     '#tyson',
     '#ubersocial',
     '#ui',
     '#ui-fail',
     '#unsix',
     '#uosxsw',
     '#usdes',
     '#usguys',
     '#uxdes',
     '#vb',
     '#vcards',
     '#vegas',
     '#verizon',
     '#veryslow',
     '#videogames',
     '#videos',
     '#view512',
     '#virtualwallet',
     '#vmware',
     '#wack',
     '#wakeuplaughing',
     '#web3',
     '#webvisions',
     '#weekend',
     '#whowillrise',
     '#win',
     '#winning',
     '#winwin',
     '#wjchat',
     '#wwsxsw',
     '#xoom',
     '#xplat',
     '#youneedthis',
     '#youtube',
     '#zaarlyiscoming',
     '#zazzlesxsw',
     '#zazzlsxsw',
     '):',
     '-->',
     '->',
     '. ...',
     '..',
     '02',
     '03',
     '0310apple',
     '1,000',
     '1,000+',
     '1.1',
     '1.6',
     '10',
     '100',
     '100s',
     '101',
     '10:30',
     '10k',
     '10mins',
     '10x',
     '11',
     '12',
     '12b',
     '12th',
     '13.6',
     '130,000',
     '14',
     '1413',
     '15',
     '150',
     '1500',
     '15k',
     '169',
     '16gb',
     '188',
     '1986',
     '1991',
     '1k',
     '1of',
     '1st',
     '20',
     '200',
     '2010',
     '2011',
     '2012/3',
     '20s',
     '22',
     '24/7',
     '25',
     '250k',
     '2:15',
     '2am',
     '2b',
     '2day',
     '2moro',
     '2nd',
     '2nite',
     '2s',
     '3.0',
     '3/13',
     '3/15',
     '3/20',
     '30',
     '300',
     '35',
     '36',
     '37',
     '3:30',
     '3d',
     '3g',
     '3gs',
     '3rd',
     '4-5',
     '4.0',
     '4.3',
     '4/5',
     '40',
     '47',
     '4am',
     '4android',
     '4chan',
     '4g',
     '4sq',
     '4square',
     '4thought',
     '5,000-',
     '5.0',
     '5.2',
     '55',
     '59',
     '5:30',
     '5hrs',
     '5pm',
     '5th',
     '6-8',
     '6.5',
     '60',
     '64g',
     '64gb',
     '64gig',
     '64mb',
     '65',
     '65.4',
     '6:30',
     '6:45',
     '6hours',
     '6th',
     '7,200',
     '7.20',
     '70',
     '75',
     '7th',
     '80',
     '800',
     '80s',
     '81',
     '82',
     '89',
     '9-15',
     '9.50',
     '90',
     '95',
     '96',
     '98.5',
     '9:30',
     ':(',
     ':)',
     ':-(',
     ':-)',
     ':-/',
     ':-d',
     ':/',
     '::',
     ':d',
     ':p',
     ';)',
     ';-)',
     '<--',
     '<---',
     '<3',
     '<amen!>',
     '<title>',
     '=d',
     '@foursquare',
     '@hamsandwich',
     '@ischafer',
     '@madebymany',
     '@malbonster',
     '@mention',
     '@partnerhub',
     '@swonderlin',
     '@sxsw',
     '@sxtxstate',
     '@wesley83',
     ']:',
     '___',
     '_µ',
     'a-ma-zing',
     'aapl',
     'abacus',
     'ability',
     'able',
     'abroad',
     'absolutely',
     'abt',
     'acceptable',
     'access',
     'accessory',
     'accommodate',
     'according',
     'account',
     'acknowledge',
     'aclu',
     'acquired',
     'action',
     'actions',
     'activations',
     'activity',
     'actsofsharing.com',
     'actual',
     'actually',
     'ad',
     'adapt',
     'add',
     'added',
     'addictive',
     'addicts',
     'adding',
     'addition',
     'additional',
     'admired',
     'admit',
     'admits',
     'ado',
     'adopter',
     'adopters',
     'adoption',
     'ads',
     'advanced',
     'advertising',
     'advice',
     'advisory',
     'affair',
     'affirmative',
     'afford',
     'afternoon',
     'age',
     'agencies',
     'agency',
     'agenda',
     'agents',
     'ago',
     'agree',
     'agreed',
     'ah',
     'ahead',
     'ahem',
     'ahh',
     'ahhh',
     'ahing',
     'aim',
     "ain't",
     'air',
     'airplane',
     'airport',
     'airports',
     'airs',
     'aka',
     'akqas',
     'al',
     'alamo',
     'alan',
     'alarm',
     'album',
     'alcoholics',
     'alex',
     'algorithm',
     'alive',
     'allow',
     'allowing',
     'allows',
     'already-dwindling',
     'alternate',
     'amazing',
     'amazingly',
     'amazon',
     'ambassador',
     'amble',
     'america',
     'amid',
     'amigos',
     'analytics',
     'andoid',
     'android',
     'angry',
     'animation',
     'announce',
     'announced',
     'announcements',
     'announces',
     'announcing',
     'annoyed',
     'anoth',
     'answer',
     'answered',
     'anti',
     'anticipate',
     'antonio',
     'antwoord',
     'anxious',
     'anybody',
     'anymore',
     'anyways',
     'ap',
     'apac',
     'apartment',
     'api',
     'apologies',
     'app',
     'apparent',
     'apparently',
     'appealing',
     'appear',
     'appears',
     'applause',
     'apple',
     "apple's",
     'apples',
     'appolicious',
     'appreciation',
     'approaches',
     'approval',
     'approved',
     'apps',
     'appstore',
     'aquent',
     'arcade',
     'archive',
     'arctic',
     'arg',
     'armed',
     'aron',
     'arrived',
     'arrives',
     'art',
     'article',
     'articles',
     'articulate',
     'artificial',
     'artist',
     'artists',
     'artwork',
     'artworks',
     'arw',
     'asddieu',
     'ask',
     'asked',
     'asking',
     'asks',
     'asleep',
     'ass',
     'assume',
     'atari',
     'atms',
     'att',
     'attend',
     'attendees',
     'attending',
     'attention',
     'attracted',
     'attracting',
     'atx',
     'atåê',
     'audience',
     'audio',
     'augmented',
     "auntie's",
     'aus',
     'austin',
     "austin's",
     'austin-area',
     'austin-bergstrom',
     'austin-bound',
     'auth',
     'authorization',
     'autistic',
     'auto',
     'auto-correct',
     'autocorrect',
     'autocorrected',
     'autocorrects',
     'available',
     'ave',
     'avenue',
     'avoid',
     'avoiding',
     'aw',
     'awards',
     'awareness',
     'away',
     'awesome',
     'awesomely',
     'awesomeness',
     'awhile',
     'awkward',
     'b',
     'b4',
     'baby',
     'back-up',
     'background',
     'backpack',
     'backup',
     'backupify',
     'bad',
     'badge',
     'badgeless',
     'badges',
     'bag',
     'bahahahaha',
     'bajillions',
     'balance',
     'ball',
     'ballroom',
     'ballrooms',
     'banality',
     'band',
     'bands',
     'bandwaggoners',
     'bandwidth',
     'bang',
     'bank',
     'banking',
     'banks',
     'bar',
     'barely',
     'barging',
     'barroom',
     'barry',
     'bars',
     'bart',
     'based',
     'basically',
     'basics',
     'basis',
     'bastards',
     'bat',
     'bathroom',
     'batteries',
     'battery',
     'battle',
     'bavc',
     'bavc.org/impact',
     'bb',
     'bbq',
     'bc',
     'beach',
     'beans',
     'bear-creatures',
     'beard',
     'beat',
     'beats',
     'beautiful',
     'beauty',
     'bed',
     'beer',
     'begin',
     'begins',
     'behave',
     'behaving',
     'behavior',
     'believe',
     'belinsky',
     'belong',
     'benefit',
     'bereft',
     'bernd',
     'best',
     'bestie',
     'bet',
     'beta',
     'better',
     'bff',
     'bicycle',
     'big',
     'bigger',
     'biggest',
     'bike',
     'billion',
     'bin',
     'bing',
     "bing's",
     'biomimicry',
     'birds',
     'birth',
     'birthday',
     'bit',
     'bit.ly/ea1zgd',
     'bit.ly/g03mzb',
     'bit.ly/i41h53',
     'biz',
     'bizzy',
     'black',
     'blackberry',
     'blame',
     'bldg',
     'block',
     'blocked',
     'blocking',
     'blocks',
     'blog',
     'bloggable',
     'blogger',
     'blogging',
     'blogs',
     'bloody',
     'bloomberg',
     'blows',
     'blue',
     'bluezoom',
     'blurs',
     'board',
     'boarded',
     'bomb',
     'bonus',
     'boo',
     'book',
     'books',
     'boom',
     'boomers',
     'boooo',
     'booth',
     'booyah',
     'bored',
     'boring',
     'borrow',
     'borrowing',
     'boss',
     'bots',
     'bought',
     'bounced',
     'bout',
     'box',
     'boxes',
     'boyfriend',
     'boyfriends',
     'boys',
     'bpm',
     'bracket',
     'brah',
     'brain',
     'brand',
     'brands',
     'brawls',
     'bread',
     ...]




```python
importance = pd.Series(ran_for.feature_importances_,index=tf_idf_vectorizer.get_feature_names())
importance
```




    ##sxsw      1.617660e-06
    #10         3.874117e-07
    #106        0.000000e+00
    #11ntc      4.550715e-04
    #1406-08    3.449528e-09
                    ...     
    ûïmute      1.380013e-05
    ûò          2.104275e-04
    ûòand       3.954520e-04
    ûó          1.350616e-04
    ûóthe       1.488186e-04
    Length: 4295, dtype: float64




```python
importance.sort_values(ascending=False).head(20).plot(kind='bar')
plt.title('Word Importance', fontsize=20)
plt.ylabel(ylabel='Frequency',fontsize=14)
plt.xlabel(xlabel='Word', fontsize=14)
plt.show()
```


![png](Tweet_Analysis_files/Tweet_Analysis_82_0.png)



```python
vectorizer = CountVectorizer()
tf_transform = TfidfTransformer(use_idf=True)
```


```python
text_pipe = Pipeline(steps=[
    ('count_vectorizer',vectorizer),
    ('tf_transformer',tf_transform)])
```


```python
RandomForestClassifier(class_weight='balanced')
```




    RandomForestClassifier(class_weight='balanced')




```python
full_pipe = Pipeline(steps=[
    ('text_pipe',text_pipe),
    ('clf',RandomForestClassifier(class_weight='balanced'))
])
```


```python
X_train_pipe = text_pipe.fit_transform(X_train)
```


```python
X_test_pipe = text_pipe.transform(X_test)
```


```python
X_train_pipe
```




    <2625x4256 sparse matrix of type '<class 'numpy.float64'>'
    	with 44273 stored elements in Compressed Sparse Row format>




```python
params = {'text_pipe__tf_transformer__use_idf':[True, False],
         'text_pipe__count_vectorizer__tokenizer':[None,tokenizer.tokenize],
         'text_pipe__count_vectorizer__stop_words':[None,stopword_list],
         'clf__criterion':['gini', 'entropy']}
```


```python
## Make and fit grid
grid = GridSearchCV(full_pipe,params,cv=3)
grid.fit(X_train,y_train)
## Display best params
grid.best_params_
```

    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [":'[", ':/', 'a-z', 'a-za-z', 'http', 'n', 'w', '‘'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    C:\Users\josep\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:383: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['http'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '
    




    {'clf__criterion': 'entropy',
     'text_pipe__count_vectorizer__stop_words': ['seems',
      'which',
      'thereupon',
      'really',
      'except',
      'ourselves',
      'third',
      'those',
      'never',
      'more',
      'whither',
      'used',
      'this',
      'two',
      '’ve',
      'whom',
      'nothing',
      'my',
      '’m',
      'at',
      'several',
      'eleven',
      'part',
      'seem',
      'when',
      'whereupon',
      'nor',
      'beforehand',
      'wherever',
      'quite',
      'thus',
      'could',
      '’ll',
      'again',
      'just',
      'these',
      'see',
      'via',
      'being',
      'over',
      'less',
      'perhaps',
      'next',
      'enough',
      'into',
      'and',
      'around',
      'rather',
      'n‘t',
      'unless',
      'moreover',
      'else',
      'sometimes',
      'his',
      'above',
      'thereby',
      'most',
      'however',
      'where',
      'amongst',
      'beyond',
      'through',
      'empty',
      'somewhere',
      'became',
      'other',
      'he',
      'same',
      'yet',
      'had',
      'become',
      'afterwards',
      'though',
      'five',
      'that',
      'was',
      'call',
      'each',
      'yourselves',
      'all',
      'then',
      'whence',
      'side',
      'go',
      'fifty',
      'twenty',
      'very',
      'per',
      'between',
      'someone',
      'still',
      'six',
      'neither',
      'keep',
      'indeed',
      'than',
      'twelve',
      '‘m',
      'against',
      'doing',
      'yours',
      'everywhere',
      'whereas',
      'down',
      'further',
      'whole',
      'of',
      'whenever',
      'top',
      'anyone',
      'to',
      'together',
      'below',
      'out',
      'within',
      'here',
      'whose',
      'hundred',
      'latterly',
      'fifteen',
      'many',
      'mostly',
      'who',
      'serious',
      'ca',
      'now',
      'beside',
      'none',
      'because',
      'in',
      'bottom',
      'ever',
      'every',
      'might',
      'former',
      'various',
      'since',
      'either',
      '‘ve',
      'once',
      'she',
      'elsewhere',
      'whoever',
      'hereby',
      'you',
      'whatever',
      'hereupon',
      'has',
      'have',
      'four',
      'ours',
      'say',
      'will',
      'seemed',
      'anyhow',
      'sixty',
      'everyone',
      'so',
      'upon',
      'why',
      'even',
      'amount',
      'cannot',
      'or',
      'although',
      'there',
      'itself',
      'give',
      'can',
      'make',
      'front',
      'were',
      'sometime',
      'always',
      'are',
      'may',
      'name',
      '‘d',
      'what',
      'namely',
      'eight',
      'after',
      'along',
      'anywhere',
      'made',
      'no',
      'latter',
      'something',
      'others',
      'forty',
      'by',
      'some',
      'three',
      'becomes',
      'becoming',
      '’s',
      'herein',
      'during',
      'but',
      'wherein',
      'without',
      'is',
      'full',
      'its',
      'almost',
      'anything',
      '‘s',
      'from',
      'using',
      'done',
      'an',
      "'ve",
      'both',
      'off',
      'therein',
      'onto',
      'somehow',
      'for',
      'the',
      'our',
      "'re",
      'own',
      'him',
      "'ll",
      'does',
      'been',
      'besides',
      'whether',
      'her',
      'before',
      'last',
      'toward',
      'show',
      'herself',
      'therefore',
      'their',
      'move',
      'nine',
      'anyway',
      'hence',
      'alone',
      'myself',
      'how',
      "n't",
      '’d',
      'throughout',
      'them',
      'a',
      'ten',
      'formerly',
      'me',
      'thru',
      'himself',
      'must',
      'up',
      "'s",
      'about',
      'on',
      'please',
      'across',
      'towards',
      'get',
      'if',
      'as',
      'regarding',
      'any',
      'would',
      'should',
      'too',
      're',
      'we',
      'nobody',
      'under',
      'noone',
      'yourself',
      'with',
      'often',
      'nowhere',
      'do',
      'while',
      'among',
      'otherwise',
      'am',
      'well',
      'back',
      'thence',
      'hereafter',
      'take',
      'not',
      'put',
      'least',
      'everything',
      'they',
      'i',
      'did',
      'it',
      "'d",
      'due',
      'whereby',
      'another',
      'already',
      'us',
      'thereafter',
      'until',
      '’re',
      'only',
      'few',
      'themselves',
      'hers',
      'n’t',
      '‘ll',
      'your',
      'also',
      'such',
      'nevertheless',
      'much',
      'be',
      'seeming',
      'meanwhile',
      'mine',
      '‘re',
      'one',
      'whereafter',
      "'m",
      'behind',
      'first',
      '!',
      '"',
      '#',
      '$',
      '%',
      '&',
      "'",
      '(',
      ')',
      '*',
      '+',
      ',',
      '-',
      '.',
      '/',
      ':',
      ';',
      '<',
      '=',
      '>',
      '?',
      '@',
      '[',
      '\\',
      ']',
      '^',
      '_',
      '`',
      '{',
      '|',
      '}',
      '~',
      'i',
      'me',
      'my',
      'myself',
      'we',
      'our',
      'ours',
      'ourselves',
      'you',
      "you're",
      "you've",
      "you'll",
      "you'd",
      'your',
      'yours',
      'yourself',
      'yourselves',
      'he',
      'him',
      'his',
      'himself',
      'she',
      "she's",
      'her',
      'hers',
      'herself',
      'it',
      "it's",
      'its',
      'itself',
      'they',
      'them',
      'their',
      'theirs',
      'themselves',
      'what',
      'which',
      'who',
      'whom',
      'this',
      'that',
      "that'll",
      'these',
      'those',
      'am',
      'is',
      'are',
      'was',
      'were',
      'be',
      'been',
      'being',
      'have',
      'has',
      'had',
      'having',
      'do',
      'does',
      'did',
      'doing',
      'a',
      'an',
      'the',
      'and',
      'but',
      'if',
      'or',
      'because',
      'as',
      'until',
      'while',
      'of',
      'at',
      'by',
      'for',
      'with',
      'about',
      'against',
      'between',
      'into',
      'through',
      'during',
      'before',
      'after',
      'above',
      'below',
      'to',
      'from',
      'up',
      'down',
      'in',
      'out',
      'on',
      'off',
      'over',
      'under',
      'again',
      'further',
      'then',
      'once',
      'here',
      'there',
      'when',
      'where',
      'why',
      'how',
      'all',
      'any',
      'both',
      'each',
      'few',
      'more',
      'most',
      'other',
      'some',
      'such',
      'no',
      'nor',
      'not',
      'only',
      'own',
      'same',
      'so',
      'than',
      'too',
      'very',
      's',
      't',
      'can',
      'will',
      'just',
      'don',
      "don't",
      'should',
      "should've",
      'now',
      'd',
      'll',
      'm',
      'o',
      're',
      've',
      'y',
      'ain',
      'aren',
      "aren't",
      'couldn',
      "couldn't",
      'didn',
      "didn't",
      'doesn',
      "doesn't",
      'hadn',
      "hadn't",
      'hasn',
      "hasn't",
      'haven',
      "haven't",
      'isn',
      "isn't",
      'ma',
      'mightn',
      "mightn't",
      'mustn',
      "mustn't",
      'needn',
      "needn't",
      'shan',
      "shan't",
      'shouldn',
      "shouldn't",
      'wasn',
      "wasn't",
      'weren',
      "weren't",
      'won',
      "won't",
      'wouldn',
      "wouldn't",
      '“',
      '”',
      '...',
      "''",
      '’',
      '``',
      'https',
      'rt',
      '\\.+',
      'sxsw',
      'mention',
      "[a-zA-Z]+\\'?s]",
      '(http[s]?://\\w*\\.\\w*/+\\w+)',
      '\\#\\w*',
      'RT [@]?\\w*:',
      '\\@\\w*',
      '\\d$',
      '^\\d',
      "([a-zA-Z]+(?:'[a-z]+)?)",
      '\\d.',
      '\\d',
      'RT',
      '^http[s]?',
      'za',
      '0',
      '1',
      '2',
      '3',
      '4',
      '5',
      '6',
      '7',
      '8',
      '9'],
     'text_pipe__count_vectorizer__tokenizer': None,
     'text_pipe__tf_transformer__use_idf': False}




```python
best_pipe = grid.best_estimator_
y_hat_test = grid.predict(X_test)
```


```python
evaluate_model(y_test,y_hat_test,X_test,best_pipe)
```

                  precision    recall  f1-score   support
    
               0       0.99      0.83      0.90       277
               1       0.93      0.99      0.96       598
    
        accuracy                           0.94       875
       macro avg       0.96      0.91      0.93       875
    weighted avg       0.94      0.94      0.94       875
    
    


![png](Tweet_Analysis_files/Tweet_Analysis_93_1.png)



```python
X_train_pipe.shape
```




    (2625, 4256)




```python
features = text_pipe.named_steps['count_vectorizer'].get_feature_names()
features[:10]
```




    ['000', '02', '03', '0310apple', '08', '10', '100', '100s', '101', '106']




```python
bigram_measures = nltk.collocations.BigramAssocMeasures()
tweet_finder = nltk.BigramCollocationFinder.from_words(clean_lemmatized_tokenz)
tweets_scored = tweet_finder.score_ngrams(bigram_measures.raw_freq)
```


```python
bigram1 = pd.DataFrame(tweets_scored, columns=['Words','Freq'])
bigram1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Words</th>
      <th>Freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(apple, store)</td>
      <td>0.006920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(social, network)</td>
      <td>0.005277</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(new, social)</td>
      <td>0.004837</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(google, launch)</td>
      <td>0.003912</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(link, google)</td>
      <td>0.003877</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42702</th>
      <td>(åç, complete)</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>42703</th>
      <td>(åçwhat, tech)</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>42704</th>
      <td>(åè, android)</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>42705</th>
      <td>(åè, ubersoc)</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>42706</th>
      <td>(ìù±g, wish)</td>
      <td>0.000012</td>
    </tr>
  </tbody>
</table>
<p>42707 rows × 2 columns</p>
</div>




```python
fig_dims = (20,8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.set(font_scale=2)
sns.set_style("darkgrid")
palette = sns.set_palette("dark")
ax = sns.barplot(x=bigram1.head(15)['Words'], y=bigram1.head(15)['Freq'], palette=palette)
ax.set(xlabel="Word Pairs",ylabel="Frequency")
plt.ticklabel_format(style='plain',axis='y')
plt.xticks(rotation=70)
plt.title('Top 15 Word Pairs by Frequency')
plt.show() 
```


![png](Tweet_Analysis_files/Tweet_Analysis_98_0.png)


## Deep NLP using Keras NN


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, optimizers
```


```python
model = 0
```


```python
y = np.asarray(df_upsampled['Positive_Bin']).astype('float32').reshape((-1,1))
X = np.asarray(df_upsampled['Tweet'])
```


```python
X.dtype
```




    dtype('O')




```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(X))
sequences = tokenizer.texts_to_sequences(X)
X = sequence.pad_sequences(sequences,maxlen=100)
```


```python
tokenizer.word_counts
```




    OrderedDict([('at', 1127),
                 ('sxsw', 3630),
                 ('tapworthy', 44),
                 ('ipad', 1213),
                 ('design', 89),
                 ('headaches', 41),
                 ('avoiding', 3),
                 ('the', 1847),
                 ('pitfalls', 3),
                 ('of', 753),
                 ('new', 357),
                 ('challenges', 3),
                 ('rt', 1000),
                 ('mention', 2312),
                 ('part', 12),
                 ('journalsim', 5),
                 ('is', 883),
                 ('support', 15),
                 ('democracy', 5),
                 ('yes', 47),
                 ('informed', 5),
                 ('populous', 5),
                 ('as', 128),
                 ('a', 983),
                 ('focus', 7),
                 ('does', 40),
                 ('not', 232),
                 ('that', 249),
                 ('newsapps', 17),
                 ('fuck', 8),
                 ('iphone', 763),
                 ('ubersocial', 23),
                 ('for', 1015),
                 ('now', 151),
                 ('in', 711),
                 ('app', 464),
                 ('store', 536),
                 ('includes', 23),
                 ('uberguide', 23),
                 ('to', 1439),
                 ('link', 1152),
                 ('2011', 27),
                 ('novelty', 19),
                 ('news', 67),
                 ('apps', 103),
                 ('fades', 21),
                 ('fast', 42),
                 ('among', 23),
                 ('digital', 37),
                 ('delegates', 19),
                 ('rule', 2),
                 ('no', 161),
                 ('more', 102),
                 ('ooing', 2),
                 ('and', 636),
                 ('ahing', 2),
                 ('over', 68),
                 ('your', 168),
                 ('we', 86),
                 ('get', 199),
                 ('it', 480),
                 ('its', 58),
                 ('big', 34),
                 ('deal', 6),
                 ('everybody', 3),
                 ('has', 138),
                 ('one', 149),
                 ('overheard', 9),
                 ('interactive', 34),
                 ('quot', 667),
                 ('arg', 4),
                 ('i', 703),
                 ('hate', 11),
                 ('want', 64),
                 ('my', 446),
                 ('blackberry', 39),
                 ('back', 48),
                 ('shocked', 3),
                 ('virtualwallet', 2),
                 ('nfc', 2),
                 ('iphone5', 2),
                 ('bc', 2),
                 ('standardization', 2),
                 ('while', 36),
                 ('android', 229),
                 ('will', 174),
                 ('have', 255),
                 ('confusion', 2),
                 ('tougher', 1),
                 ('crowd', 19),
                 ('than', 71),
                 ('colin', 1),
                 ('quinn', 1),
                 ('hey', 41),
                 ('marissa', 56),
                 ('mayer', 73),
                 ('please', 13),
                 ('tell', 21),
                 ('us', 61),
                 ('something', 20),
                 ('about', 133),
                 ('products', 27),
                 ('google', 801),
                 ('launched', 21),
                 ('months', 9),
                 ('ago', 7),
                 ('why', 81),
                 ('wifi', 28),
                 ('working', 26),
                 ('on', 555),
                 ('laptop', 32),
                 ('but', 160),
                 ('neither', 3),
                 ('nor', 3),
                 ('3g', 31),
                 ('grrr', 7),
                 ('starting', 7),
                 ('think', 81),
                 ('like', 165),
                 ('abacus', 4),
                 ('phones', 15),
                 ('damn', 5),
                 ('u', 34),
                 ('just', 294),
                 ('wanted', 12),
                 ('dropped', 5),
                 ('real', 11),
                 ('estate', 2),
                 ('search', 40),
                 ('b', 15),
                 ('c', 9),
                 ('do', 80),
                 ('right', 37),
                 ("they'd", 3),
                 ('need', 110),
                 ('invest', 2),
                 ('much', 57),
                 ("they're", 27),
                 ('willing', 5),
                 ('moment', 9),
                 ('\x89÷¼', 2),
                 ('are', 181),
                 ('better', 43),
                 ('\x89÷', 2),
                 ('\x89ã', 2),
                 ('edchat', 2),
                 ('musedchat', 2),
                 ('sxswi', 98),
                 ('classical', 2),
                 ('newtwitter', 2),
                 ('rumor', 36),
                 ('mill', 4),
                 ('3', 52),
                 ('between', 8),
                 ('6', 21),
                 ('15', 14),
                 ('cameras', 10),
                 ('slightly', 4),
                 ('thinner', 4),
                 ('rare', 4),
                 ('earths', 4),
                 ('case', 41),
                 ('different', 8),
                 ('still', 45),
                 ('smudgy', 4),
                 ('screen', 18),
                 ('hmm', 10),
                 ("can't", 42),
                 ('twitter', 32),
                 ('searches', 2),
                 ('update', 20),
                 ('hootsuite', 6),
                 ('or', 85),
                 ('tweetdeck', 5),
                 ('come', 71),
                 ('apple', 924),
                 ('cant', 9),
                 ('spell', 3),
                 ('checking', 9),
                 ('without', 20),
                 ('auto', 5),
                 ('correct', 4),
                 ('must', 29),
                 ('be', 263),
                 ('all', 142),
                 ('nothing', 23),
                 ('you', 367),
                 ('finally', 18),
                 ('everyone', 47),
                 ('buy', 47),
                 ('facebook', 17),
                 ('then', 18),
                 ('introduces', 3),
                 ('circles', 156),
                 ('fair', 3),
                 ('stop', 18),
                 ('with', 351),
                 ('innovation', 10),
                 ('people', 113),
                 ('expensive', 2),
                 ('mobile', 128),
                 ('data', 20),
                 ('plans', 9),
                 ('killing', 9),
                 ('flavor', 2),
                 ('contextual', 5),
                 ('discovery', 10),
                 ('abroad', 2),
                 ('pnid', 13),
                 ("there's", 24),
                 ('reason', 6),
                 ("isn't", 23),
                 ('social', 154),
                 ('they', 148),
                 ('too', 94),
                 ('technical', 8),
                 ('comments', 4),
                 ('yet', 23),
                 ('walk', 7),
                 ('into', 25),
                 ('conference', 7),
                 ('room', 19),
                 ('where', 29),
                 ("doesn't", 35),
                 ('look', 39),
                 ('an', 434),
                 ('ad', 9),
                 ("you'd", 4),
                 ('there', 94),
                 ('was', 132),
                 ('else', 15),
                 ('best', 79),
                 ('thing', 51),
                 ("i've", 56),
                 ('heard', 72),
                 ('this', 258),
                 ('weekend', 35),
                 ('gave', 33),
                 ('2', 558),
                 ('money', 45),
                 ('japan', 44),
                 ('relief', 24),
                 ("don't", 100),
                 ('seems', 30),
                 ('sabotaged', 4),
                 ('youtube', 10),
                 ('account', 8),
                 ('wtf', 5),
                 ('trying', 19),
                 ('own', 19),
                 ('entire', 11),
                 ('online', 9),
                 ('ecosystem', 4),
                 ('very', 50),
                 ('bad', 27),
                 ('form', 9),
                 ('wait', 58),
                 ('give', 25),
                 ('samsung', 6),
                 ('demo', 31),
                 ('horrible', 6),
                 ('terrible', 9),
                 ('nexus', 15),
                 ('s', 26),
                 ('phone', 58),
                 ('major', 76),
                 ('flaw', 4),
                 ('go', 103),
                 ('stay', 14),
                 ('open', 53),
                 ('when', 66),
                 ('switch', 6),
                 ('ipaddesignheadaches', 4),
                 ('nyt', 3),
                 ("here's", 9),
                 ('amazing', 29),
                 ('way', 50),
                 ('serve', 6),
                 ('our', 71),
                 ('readership', 3),
                 ('market', 30),
                 ('opportunity', 3),
                 ('ignore', 3),
                 ('crashed', 3),
                 ('amp', 208),
                 ('had', 50),
                 ('fresh', 3),
                 ('restore', 2),
                 ('lost', 16),
                 ('fave', 6),
                 ('dali', 2),
                 ('canvas', 4),
                 ('pak', 2),
                 ('can', 134),
                 ('ever', 72),
                 ('also', 53),
                 ('\x89ûï', 66),
                 ('comes', 43),
                 ('up', 495),
                 ('cool', 112),
                 ('technology', 61),
                 ("one's", 32),
                 ('because', 86),
                 ('conferences', 30),
                 ('\x89û\x9d', 8),
                 ('ridiculous', 17),
                 ('see', 92),
                 ('someone', 34),
                 ('taking', 29),
                 ('photo', 22),
                 ('during', 53),
                 ('session', 31),
                 ('their', 110),
                 ('cannot', 8),
                 ('concert', 10),
                 ('use', 72),
                 ('silly', 14),
                 ('am', 30),
                 ('grateful', 2),
                 ('bicycle', 2),
                 ('having', 25),
                 ('cursing', 2),
                 ('losing', 2),
                 ('hour', 23),
                 ('zzzs', 2),
                 ('battery', 45),
                 ('life', 38),
                 ('pop', 192),
                 ('out', 213),
                 ('ipad2s', 10),
                 ('day', 115),
                 ('1', 58),
                 ('charger', 21),
                 ('kicked', 1),
                 ('bucket', 1),
                 ('heck', 5),
                 ("that's", 28),
                 ('within', 5),
                 ('walking', 10),
                 ('distance', 1),
                 ('world', 31),
                 ('really', 64),
                 ('needs', 25),
                 ('fb', 42),
                 ('implement', 6),
                 ('eventually', 6),
                 ('clearly', 9),
                 ('another', 26),
                 ('daylight', 5),
                 ('savings', 5),
                 ('time', 128),
                 ('bug', 5),
                 ('4', 69),
                 ('alarm', 15),
                 ('remember', 9),
                 ('fix', 13),
                 ("room's", 5),
                 ('clocks', 6),
                 ('whoops', 5),
                 ('panel', 50),
                 ('staying', 6),
                 ('alive', 6),
                 ('indie', 5),
                 ('game', 46),
                 ('development', 10),
                 ('survive', 11),
                 ('kind', 7),
                 ('downer', 5),
                 ('should', 60),
                 ('try', 13),
                 ('coronasdk', 5),
                 ('if', 110),
                 ('popup', 74),
                 ('austin', 321),
                 ('sold', 22),
                 ('extenders', 3),
                 ('would', 82),
                 ('make', 44),
                 ('so', 198),
                 ('effing', 3),
                 ('hubby', 3),
                 ('line', 135),
                 ('point', 10),
                 ('him', 10),
                 ('towards', 5),
                 ('wife', 4),
                 ('number', 6),
                 ('tomorrow', 30),
                 ('sigh', 9),
                 ('guess', 18),
                 ('mom', 6),
                 ('101', 5),
                 ('class', 3),
                 ('talk', 44),
                 ('assume', 3),
                 ("didn't", 29),
                 ('ditch', 3),
                 ('previous', 3),
                 ('experience', 13),
                 ('party', 113),
                 ('kinda', 12),
                 ('embarrassed', 4),
                 ('by', 210),
                 ('row', 2),
                 ('morning', 24),
                 ('two', 29),
                 ('hand', 24),
                 ('writing', 3),
                 ('notes', 14),
                 ('person', 29),
                 ('me', 164),
                 ('pc', 7),
                 ('proof', 3),
                 ('monopoly', 5),
                 ('lt', 28),
                 ('amen', 2),
                 ('gt', 41),
                 ('hear', 8),
                 ('preferrably', 2),
                 ('hashtags', 3),
                 ('deficit', 2),
                 ('alarms', 1),
                 ('botch', 1),
                 ('timechange', 1),
                 ('how', 110),
                 ('many', 49),
                 ("sxsw'ers", 1),
                 ('freak', 1),
                 ('late', 26),
                 ('flights', 5),
                 ('missed', 4),
                 ('panels', 8),
                 ('behind', 15),
                 ('bloody', 5),
                 ('marys', 1),
                 ('rip', 2),
                 ('june', 5),
                 ('2010', 2),
                 ('survived', 3),
                 ('severe', 2),
                 ('drop', 6),
                 ('could', 40),
                 ('evade', 2),
                 ('drowning', 3),
                 ('heading', 28),
                 ('hilton', 8),
                 ('salon', 8),
                 ('j', 8),
                 ('hmmm', 4),
                 ('taxi', 6),
                 ('magic', 9),
                 ('appear', 3),
                 ('any', 35),
                 ('das', 2),
                 ('verpixelungsrecht\x89ûóthe', 2),
                 ('house', 12),
                 ('pixelated', 2),
                 ('street', 16),
                 ('view\x89ûó', 2),
                 ('theft', 2),
                 ('from', 195),
                 ('public', 6),
                 ('classiest', 13),
                 ('fascist', 20),
                 ('company', 32),
                 ('america', 22),
                 ('elegant', 12),
                 ('rji', 8),
                 ('glow', 5),
                 ('dark', 7),
                 ('cup', 2),
                 ('leaked', 2),
                 ('goo', 2),
                 ('camera', 15),
                 ('bag', 8),
                 ('source', 11),
                 ('code', 13),
                 ('even', 73),
                 ('longer', 6),
                 ('take', 28),
                 ('friends', 24),
                 ('changed', 6),
                 ('instead', 24),
                 ('forward', 21),
                 ('t', 43),
                 ('hints', 1),
                 ('help', 15),
                 ('jeez', 2),
                 ('guys', 23),
                 ('dunno', 2),
                 ("gold's", 5),
                 ('gym', 9),
                 ('realize', 2),
                 ('un', 3),
                 ('jobs', 8),
                 ('aesthetic', 2),
                 ('awesome', 87),
                 ('acknowledge', 1),
                 ('maps', 96),
                 ('increase', 1),
                 ('product', 28),
                 ('incorrect', 1),
                 ('routes', 6),
                 ('fail', 34),
                 ('okay', 5),
                 ('debuting', 5),
                 ('today', 132),
                 ('dear', 10),
                 ('suck', 8),
                 ('again', 25),
                 ('year', 55),
                 ('sitby', 1),
                 ('great', 114),
                 ('include', 1),
                 ('film', 4),
                 ('sessions', 13),
                 ('hope', 19),
                 ("year's", 12),
                 ('festival', 15),
                 ('crashy', 4),
                 ('brought', 8),
                 ('rerouted', 5),
                 ('images', 10),
                 ('jcpenney', 5),
                 ('macys', 5),
                 ('trashy', 5),
                 ('restraunts', 5),
                 ('comment', 5),
                 ('uosxsw', 8),
                 ('guy', 38),
                 ('explaining', 7),
                 ('he', 21),
                 ('made', 48),
                 ('realistic', 7),
                 ('bots', 7),
                 ('experiment', 10),
                 ('gee', 7),
                 ('thanks', 80),
                 ('doing', 47),
                 ('via', 139),
                 ('shop', 35),
                 ('sucks', 18),
                 ('toast', 3),
                 ('convince', 5),
                 ('users', 77),
                 ('start', 16),
                 ('interface', 19),
                 ("you'll", 9),
                 ('5', 37),
                 ('weeks', 8),
                 ('appleatxdt', 6),
                 ('shipments', 4),
                 ('daily', 13),
                 ('probably', 22),
                 ('put', 14),
                 ('away', 33),
                 ('dailies', 2),
                 ('going', 76),
                 ('evolve', 2),
                 ('impossible', 3),
                 ('download', 31),
                 ('20', 8),
                 ('mb', 2),
                 ('downloads', 5),
                 ('getting', 50),
                 ('panned', 3),
                 ('trumping', 3),
                 ('content', 39),
                 ('rightfully', 3),
                 ('looks', 45),
                 ('stupid', 18),
                 ('pix', 3),
                 ('believe', 19),
                 ('looking', 29),
                 ('release', 28),
                 ('native', 5),
                 ('0', 10),
                 ('tablet', 17),
                 ('optimized', 3),
                 ('clients', 3),
                 ('latitude', 4),
                 ('totalitarian', 2),
                 ('thought', 14),
                 ('action', 25),
                 ('worldwide', 2),
                 ('tx', 7),
                 ('decide', 2),
                 ('spend', 4),
                 ('barry', 17),
                 ('diller', 25),
                 ('says', 39),
                 ("you're", 25),
                 ('write', 12),
                 ('factor', 6),
                 ('only', 87),
                 ('fully', 4),
                 ('anticipate', 3),
                 ('every', 48),
                 ('geek', 21),
                 ('douche', 8),
                 ('toting', 4),
                 ('these', 17),
                 ("it's", 148),
                 ('worth', 17),
                 ('know', 49),
                 ('week', 38),
                 ('tough', 5),
                 ('already', 54),
                 ('dwindling', 4),
                 ('jeebus', 4),
                 ('keep', 22),
                 ('correcting', 4),
                 ('curse', 4),
                 ('words', 5),
                 ('cab', 8),
                 ('ride', 4),
                 ('hell', 4),
                 ('mall', 6),
                 ('were', 24),
                 ('dinner', 4),
                 ('figure', 11),
                 ('hotel', 10),
                 ('prefers', 1),
                 ('launch', 184),
                 ('hyped', 1),
                 ('features', 13),
                 ('meh', 1),
                 ('bang', 1),
                 ('tc', 5),
                 ('anyone', 21),
                 ('who', 33),
                 ('donate', 3),
                 ('tsunami', 5),
                 ('victims', 3),
                 ('enough', 7),
                 ('network', 97),
                 ('called', 80),
                 ('possibly', 49),
                 ('sxsw\x89û\x9d', 34),
                 ('after', 31),
                 ('hrs', 7),
                 ('tweeting', 17),
                 ('rise', 8),
                 ('dead', 8),
                 ('upgrade', 11),
                 ('plugin', 7),
                 ('stations', 7),
                 ('flipboard', 24),
                 ('bldg', 4),
                 ('appstore', 5),
                 ('wholistic', 4),
                 ('mktg', 6),
                 ('awareness', 4),
                 ('p', 11),
                 ('r', 9),
                 ('strategy', 11),
                 ('drive', 8),
                 ('adoption', 8),
                 ('rely', 4),
                 ('meant', 7),
                 ('wish', 27),
                 ('dyac', 5),
                 ('tomlinson', 5),
                 ('observer', 3),
                 ('subscription', 2),
                 ('holding', 7),
                 ('biggest', 4),
                 ('impediment', 2),
                 ('success', 19),
                 ('crazy', 7),
                 ("80's", 4),
                 ('cell', 1),
                 ('blocks', 12),
                 ('long', 39),
                 ('betterthingstodo', 1),
                 ('down', 38),
                 ('idiot', 2),
                 ('smcomedyfyeah', 2),
                 ('teeming', 4),
                 ('sea', 5),
                 ('addicts', 5),
                 ('busy', 10),
                 ('twittering', 4),
                 ('engage', 4),
                 ('anoth\x89û', 4),
                 ('cont', 11),
                 ('cake', 2),
                 ('around', 63),
                 ('tries', 4),
                 ('310409h2011', 2),
                 ('deleting', 6),
                 ('full', 27),
                 ('geeks', 13),
                 ('talking', 8),
                 ('tv', 21),
                 ('asked', 7),
                 ('uses', 9),
                 ('raised', 2),
                 ('socialviewing', 4),
                 ('next', 70),
                 ('media', 20),
                 ('flop', 5),
                 ('preview', 10),
                 ('service', 20),
                 ('caring', 11),
                 ('business', 18),
                 ('tim', 9),
                 ("o'reilly", 8),
                 ("google's", 41),
                 ('remove', 3),
                 ('deadly', 3),
                 ('such', 8),
                 ('through', 39),
                 ('death', 3),
                 ('valley', 4),
                 ('likes', 7),
                 ('pay', 6),
                 ('them', 42),
                 ('what', 118),
                 ('thinks', 5),
                 ('nuts', 5),
                 ('worst', 5),
                 ('droid', 16),
                 ('spontaniety', 4),
                 ('been', 37),
                 ('replaced', 4),
                 ('dies', 4),
                 ('home', 20),
                 ('vs', 32),
                 ('bing', 37),
                 ('bettersearch', 6),
                 ('shot', 12),
                 ('w', 90),
                 ('structured', 5),
                 ('potentially', 5),
                 ('higher', 8),
                 ('margin', 5),
                 ('cpa', 5),
                 ('model', 6),
                 ("'s", 12),
                 ('10x', 10),
                 ('useful', 20),
                 ('iphone4', 11),
                 ('ppl', 21),
                 ('noticed', 5),
                 ('dst', 5),
                 ('coming', 55),
                 ('sunday', 8),
                 ('insane', 5),
                 ('navigating', 5),
                 ('crowded', 11),
                 ('worse', 3),
                 ('walks', 4),
                 ('face', 8),
                 ('adpeopleproblems', 3),
                 ('fed', 3),
                 ('angrybirds', 3),
                 ('julian', 3),
                 ('screamed', 3),
                 ('got', 85),
                 ('fly', 3),
                 ('zone', 3),
                 ('here', 99),
                 ('pigfucker', 3),
                 ('threw', 3),
                 ('his', 33),
                 ('some', 47),
                 ('kid', 9),
                 ('deliciously', 4),
                 ('ironic', 11),
                 ('privacy', 8),
                 ('whole', 13),
                 ('banking', 6),
                 ('cartel', 4),
                 ('military', 4),
                 ('scientific', 4),
                 ('dictatorship', 4),
                 ('takeover', 4),
                 ('rant', 4),
                 ('sxxpress', 2),
                 ('almost', 11),
                 ('patented', 3),
                 ('e', 6),
                 ('age', 5),
                 ('domain', 4),
                 ('rankings', 9),
                 ('algorithm', 4),
                 ('mean', 11),
                 ('qagb', 16),
                 ('delicious', 6),
                 ('4g', 5),
                 ('struggle', 6),
                 ('anything', 19),
                 ('well', 35),
                 ('cashmore', 8),
                 ('crushing', 1),
                 ('non', 8),
                 ('endorsement', 2),
                 ('checkins', 9),
                 ('ouch', 3),
                 ('popular', 6),
                 ('kids', 10),
                 ('concept', 7),
                 ('anyway', 8),
                 ('true', 12),
                 ("'google", 4),
                 ("users'", 7),
                 ('psych', 5),
                 ('burn', 7),
                 ('eg', 1),
                 ('v', 8),
                 ('bored', 7),
                 ('find', 18),
                 ('stream', 18),
                 ('follow', 12),
                 ('inane', 2),
                 ('tweets', 7),
                 ('surely', 7),
                 ('southby', 3),
                 ('mistakes', 18),
                 ('building', 23),
                 ('plus', 12),
                 ('page', 12),
                 ('rank', 3),
                 ('ridiculously', 4),
                 ('al', 3),
                 ('franken', 3),
                 ('justin', 3),
                 ('timberlake', 3),
                 ('95', 2),
                 ('less', 5),
                 ('000', 9),
                 ('total', 6),
                 ('def', 8),
                 ('sorta', 3),
                 ('pretty', 29),
                 ('sux', 3),
                 ('protip', 2),
                 ('avoid', 2),
                 ('area', 1),
                 ('stores', 6),
                 ('friday', 8),
                 ('steve', 10),
                 ('position', 4),
                 ('device', 15),
                 ('china', 4),
                 ('suicide', 4),
                 ('rates', 4),
                 ('high', 14),
                 ('sells', 4),
                 ('dreams', 4),
                 ('kawasaki', 16),
                 ('thisisdare', 4),
                 ('old', 23),
                 ('schedule', 16),
                 ('oh', 39),
                 ('noes', 4),
                 ('rejection', 4),
                 ('stopped', 3),
                 ('lame', 13),
                 ('z', 3),
                 ('cut', 5),
                 ('weight', 3),
                 ('half', 13),
                 ('metaphor', 7),
                 ('book', 31),
                 ('behave', 5),
                 ('simple', 8),
                 ('stuff', 25),
                 ('often', 6),
                 ('forgotten', 6),
                 ('diabetes', 2),
                 ('plate', 2),
                 ('covered', 8),
                 ('josh', 13),
                 ('clark', 8),
                 ("ipad's", 13),
                 ('button', 13),
                 ('heat', 8),
                 ('million', 20),
                 ('suns', 6),
                 ('tablets', 9),
                 ('xoom', 11),
                 ('touch', 9),
                 ('emulates', 3),
                 ('mouse', 3),
                 ('click', 3),
                 ('keyboard', 7),
                 ('input', 3),
                 ('means', 12),
                 ("we're", 36),
                 ('gswsxsw', 4),
                 ('futureoftouch', 3),
                 ('com', 32),
                 ('live', 40),
                 ('compatible', 5),
                 ('maybe', 17),
                 ('sitting', 8),
                 ('floor', 4),
                 ("who's", 13),
                 ('fondling', 3),
                 ('disturbing', 3),
                 ('graph', 2),
                 ('did', 15),
                 ('repair', 1),
                 ('damage', 1),
                 ('title', 3),
                 ('tag', 3),
                 ('microformats', 1),
                 ('waze', 3),
                 ('duking', 3),
                 ('re', 12),
                 ('personalized', 12),
                 ('mapping', 4),
                 ('friendly', 4),
                 ('crashing', 5),
                 ('lunch', 9),
                 ('cnngrill', 8),
                 ('view', 15),
                 ('html5', 13),
                 ('dev', 16),
                 ('trenches', 6),
                 ('painful', 6),
                 ('ios', 26),
                 ('sleek', 6),
                 ('fucking', 10),
                 ('mac', 22),
                 ('cwebb', 5),
                 ('grant', 5),
                 ('hill', 5),
                 ('respectfully', 3),
                 ('disagree', 3),
                 ('problem', 8),
                 ('ubiquitous', 3),
                 ('project314', 3),
                 ('speaks', 2),
                 ('truth', 2),
                 ('watched', 4),
                 ('staff', 2),
                 ('temp', 40),
                 ('five', 6),
                 ('facepalmed', 2),
                 ('ugh', 2),
                 ('longlinesbadux', 2),
                 ("won't", 22),
                 ('heads', 9),
                 ('gps', 10),
                 ('messed', 2),
                 ("i'm", 125),
                 ('yonkers', 2),
                 ('good', 106),
                 ('hotpot', 22),
                 ('rate', 5),
                 ('restaurants', 3),
                 ('recos', 3),
                 ('eat', 4),
                 ('um', 3),
                 ('foursquare', 18),
                 ('yelp', 8),
                 ('etc', 9),
                 ('queue', 13),
                 ('itunes', 18),
                 ('work', 46),
                 ('ie', 3),
                 ('run', 13),
                 ('software', 8),
                 ('ubuntu', 2),
                 ('desktop', 2),
                 ('dl', 3),
                 ('presentation', 28),
                 ('sales', 11),
                 ('pitch', 5),
                 ('mozilla', 2),
                 ('crapkit', 2),
                 ('webkit', 2),
                 ('gecko', 2),
                 ('bandwaggoners', 2),
                 ('rear', 1),
                 ('facing', 1),
                 ('found', 10),
                 ('kyping', 5),
                 ("iphone's", 12),
                 ('geolocation', 7),
                 ('releasing', 6),
                 ('background', 5),
                 ('patch', 5),
                 ('batterykiller', 7),
                 ('learn', 12),
                 ('step', 9),
                 ('becoming', 3),
                 ('skynet', 2),
                 ('never', 22),
                 ('apparent', 2),
                 ('nice', 47),
                 ('removable', 2),
                 ('batteries', 2),
                 ('alwayshavingtoplugin', 2),
                 ('gsdm', 24),
                 ('somebody', 5),
                 ('lets', 9),
                 ...])




```python
vocab_size = len(tokenizer.word_counts)
seq_len = X.shape[1]
```


```python
vocab_size
seq_len
```




    100




```python
print(type(X),X.shape)
print(type(y),y.shape)
```

    <class 'numpy.ndarray'> (3500, 100)
    <class 'numpy.ndarray'> (3500, 1)
    


```python
X = np.asarray(X).astype('float32')
```


```python
print(type(X),X.shape)
print(type(y),y.shape)
```

    <class 'numpy.ndarray'> (3500, 100)
    <class 'numpy.ndarray'> (3500, 1)
    


```python
# X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=42, test_size=.2)
```


```python
def create_model(vocab_size,seq_len):
    
    model = Sequential()
    embedding_size = 128
    model.add(Embedding(vocab_size, seq_len, input_length=seq_len))
    model.add(Dense(16,input_dim=2, activation='relu'))
    model.add(LSTM(8,input_dim=2, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['precision'])
    
    model.summary()
    
    return model
```


```python
model = create_model(vocab_size,seq_len)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 100, 100)          481600    
    _________________________________________________________________
    dense (Dense)                (None, 100, 16)           1616      
    _________________________________________________________________
    lstm (LSTM)                  (None, 8)                 800       
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 18        
    =================================================================
    Total params: 484,034
    Trainable params: 484,034
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.fit(X,y, batch_size=32, epochs=5, verbose=1, validation_split=.2)
```

    Epoch 1/5
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-130-c1e6834d6817> in <module>
    ----> 1 model.fit(X,y, batch_size=32, epochs=5, verbose=1, validation_split=.2)
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1098                 _r=1):
       1099               callbacks.on_train_batch_begin(step)
    -> 1100               tmp_logs = self.train_function(iterator)
       1101               if data_handler.should_sync:
       1102                 context.async_wait()
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\eager\def_function.py in __call__(self, *args, **kwds)
        826     tracing_count = self.experimental_get_tracing_count()
        827     with trace.Trace(self._name) as tm:
    --> 828       result = self._call(*args, **kwds)
        829       compiler = "xla" if self._experimental_compile else "nonXla"
        830       new_tracing_count = self.experimental_get_tracing_count()
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\eager\def_function.py in _call(self, *args, **kwds)
        869       # This is the first call of __call__, so we have to initialize.
        870       initializers = []
    --> 871       self._initialize(args, kwds, add_initializers_to=initializers)
        872     finally:
        873       # At this point we know that the initialization is complete (or less
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\eager\def_function.py in _initialize(self, args, kwds, add_initializers_to)
        723     self._graph_deleter = FunctionDeleter(self._lifted_initializer_graph)
        724     self._concrete_stateful_fn = (
    --> 725         self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
        726             *args, **kwds))
        727 
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\eager\function.py in _get_concrete_function_internal_garbage_collected(self, *args, **kwargs)
       2967       args, kwargs = None, None
       2968     with self._lock:
    -> 2969       graph_function, _ = self._maybe_define_function(args, kwargs)
       2970     return graph_function
       2971 
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\eager\function.py in _maybe_define_function(self, args, kwargs)
       3359 
       3360           self._function_cache.missed.add(call_context_key)
    -> 3361           graph_function = self._create_graph_function(args, kwargs)
       3362           self._function_cache.primary[cache_key] = graph_function
       3363 
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\eager\function.py in _create_graph_function(self, args, kwargs, override_flat_arg_shapes)
       3194     arg_names = base_arg_names + missing_arg_names
       3195     graph_function = ConcreteFunction(
    -> 3196         func_graph_module.func_graph_from_py_func(
       3197             self._name,
       3198             self._python_function,
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\framework\func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, autograph, autograph_options, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, override_flat_arg_shapes)
        988         _, original_func = tf_decorator.unwrap(python_func)
        989 
    --> 990       func_outputs = python_func(*func_args, **func_kwargs)
        991 
        992       # invariant: `func_outputs` contains only Tensors, CompositeTensors,
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\eager\def_function.py in wrapped_fn(*args, **kwds)
        632             xla_context.Exit()
        633         else:
    --> 634           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
        635         return out
        636 
    

    ~\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\framework\func_graph.py in wrapper(*args, **kwargs)
        975           except Exception as e:  # pylint:disable=broad-except
        976             if hasattr(e, "ag_error_metadata"):
    --> 977               raise e.ag_error_metadata.to_exception(e)
        978             else:
        979               raise
    

    ValueError: in user code:
    
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\engine\training.py:805 train_function  *
            return step_function(self, iterator)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\engine\training.py:795 step_function  **
            outputs = model.distribute_strategy.run(run_step, args=(data,))
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\distribute\distribute_lib.py:1259 run
            return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\distribute\distribute_lib.py:2730 call_for_each_replica
            return self._call_for_each_replica(fn, args, kwargs)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\distribute\distribute_lib.py:3417 _call_for_each_replica
            return fn(*args, **kwargs)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\engine\training.py:788 run_step  **
            outputs = model.train_step(data)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\engine\training.py:755 train_step
            loss = self.compiled_loss(
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\engine\compile_utils.py:203 __call__
            loss_value = loss_obj(y_t, y_p, sample_weight=sw)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\losses.py:152 __call__
            losses = call_fn(y_true, y_pred)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\losses.py:256 call  **
            return ag_fn(y_true, y_pred, **self._fn_kwargs)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\util\dispatch.py:201 wrapper
            return target(*args, **kwargs)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\losses.py:1608 binary_crossentropy
            K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\util\dispatch.py:201 wrapper
            return target(*args, **kwargs)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\keras\backend.py:4979 binary_crossentropy
            return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\util\dispatch.py:201 wrapper
            return target(*args, **kwargs)
        C:\Users\josep\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\ops\nn_impl.py:173 sigmoid_cross_entropy_with_logits
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
    
        ValueError: logits and labels must have the same shape ((None, 2) vs (None, 1))
    



```python

```

## Deep NLP using Word2Vec NN


```python
from nltk import word_tokenize
```


```python
data = df_upsampled['Tweet'].map(word_tokenize)
```


```python
data[:10]
```




    1749    [At, #, sxsw, #, tapworthy, iPad, Design, Head...
    6436    [RT, @, mention, Part, of, Journalsim, is, the...
    3838    [Fuck, the, iphone, !, RT, @, mention, New, #,...
    1770    [#, SXSW, 2011, :, Novelty, of, iPad, news, ap...
    1062    [New, #, SXSW, rule, :, no, more, ooing, and, ...
    324     [Overheard, at, #, sxsw, interactive, :, &, qu...
    1944    [#, virtualwallet, #, sxsw, no, NFC, in, #, ip...
    7201    [#, SXSW, a, tougher, crowd, than, Colin, Quin...
    3159    [Why, is, wifi, working, on, my, laptop, but, ...
    4631    [Is, starting, to, think, my, #, blackberry, i...
    Name: Tweet, dtype: object




```python
model_W2V = Word2Vec(data, size =100, window=5, min_count=1, workers=4)
```

    2020-12-17 13:06:20,152 : INFO : collecting all words and their counts
    2020-12-17 13:06:20,153 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2020-12-17 13:06:20,171 : INFO : collected 5920 word types from a corpus of 86715 raw words and 3500 sentences
    2020-12-17 13:06:20,172 : INFO : Loading a fresh vocabulary
    2020-12-17 13:06:20,184 : INFO : effective_min_count=1 retains 5920 unique words (100% of original 5920, drops 0)
    2020-12-17 13:06:20,185 : INFO : effective_min_count=1 leaves 86715 word corpus (100% of original 86715, drops 0)
    2020-12-17 13:06:20,204 : INFO : deleting the raw counts dictionary of 5920 items
    2020-12-17 13:06:20,207 : INFO : sample=0.001 downsamples 52 most-common words
    2020-12-17 13:06:20,208 : INFO : downsampling leaves estimated 56808 word corpus (65.5% of prior 86715)
    2020-12-17 13:06:20,219 : INFO : estimated required memory for 5920 words and 100 dimensions: 7696000 bytes
    2020-12-17 13:06:20,220 : INFO : resetting layer weights
    2020-12-17 13:06:21,308 : INFO : training model with 4 workers on 5920 vocabulary and 100 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2020-12-17 13:06:21,345 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,347 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,351 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,352 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,354 : INFO : EPOCH - 1 : training on 86715 raw words (56795 effective words) took 0.0s, 1451503 effective words/s
    2020-12-17 13:06:21,397 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,402 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,404 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,405 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,407 : INFO : EPOCH - 2 : training on 86715 raw words (56775 effective words) took 0.0s, 1289936 effective words/s
    2020-12-17 13:06:21,450 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,458 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,459 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,460 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,460 : INFO : EPOCH - 3 : training on 86715 raw words (56788 effective words) took 0.0s, 1323415 effective words/s
    2020-12-17 13:06:21,504 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,508 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,509 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,510 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,510 : INFO : EPOCH - 4 : training on 86715 raw words (56940 effective words) took 0.0s, 1330629 effective words/s
    2020-12-17 13:06:21,553 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,559 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,561 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,562 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,563 : INFO : EPOCH - 5 : training on 86715 raw words (56768 effective words) took 0.0s, 1218234 effective words/s
    2020-12-17 13:06:21,563 : INFO : training on a 433575 raw words (284066 effective words) took 0.3s, 1116329 effective words/s
    


```python
model_W2V.train(data,total_examples=model_W2V.corpus_count, epochs=10)
```

    2020-12-17 13:06:21,569 : WARNING : Effective 'alpha' higher than previous training cycles
    2020-12-17 13:06:21,570 : INFO : training model with 4 workers on 5920 vocabulary and 100 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2020-12-17 13:06:21,620 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,626 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,629 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,630 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,631 : INFO : EPOCH - 1 : training on 86715 raw words (56706 effective words) took 0.1s, 1080911 effective words/s
    2020-12-17 13:06:21,672 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,678 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,679 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,681 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,682 : INFO : EPOCH - 2 : training on 86715 raw words (56852 effective words) took 0.0s, 1246815 effective words/s
    2020-12-17 13:06:21,724 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,727 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,730 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,732 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,733 : INFO : EPOCH - 3 : training on 86715 raw words (56891 effective words) took 0.0s, 1274723 effective words/s
    2020-12-17 13:06:21,769 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,772 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,777 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,778 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,778 : INFO : EPOCH - 4 : training on 86715 raw words (56852 effective words) took 0.0s, 1423043 effective words/s
    2020-12-17 13:06:21,819 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,826 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,828 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,830 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,831 : INFO : EPOCH - 5 : training on 86715 raw words (56672 effective words) took 0.0s, 1213146 effective words/s
    2020-12-17 13:06:21,876 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,882 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,883 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,884 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,885 : INFO : EPOCH - 6 : training on 86715 raw words (56801 effective words) took 0.0s, 1174244 effective words/s
    2020-12-17 13:06:21,921 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,929 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,931 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,932 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,932 : INFO : EPOCH - 7 : training on 86715 raw words (56838 effective words) took 0.0s, 1417548 effective words/s
    2020-12-17 13:06:21,969 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:21,975 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:21,978 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:21,979 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:21,979 : INFO : EPOCH - 8 : training on 86715 raw words (56914 effective words) took 0.0s, 1381844 effective words/s
    2020-12-17 13:06:22,019 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:22,020 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:22,024 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:22,026 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:22,028 : INFO : EPOCH - 9 : training on 86715 raw words (56831 effective words) took 0.0s, 1314607 effective words/s
    2020-12-17 13:06:22,066 : INFO : worker thread finished; awaiting finish of 3 more threads
    2020-12-17 13:06:22,074 : INFO : worker thread finished; awaiting finish of 2 more threads
    2020-12-17 13:06:22,075 : INFO : worker thread finished; awaiting finish of 1 more threads
    2020-12-17 13:06:22,076 : INFO : worker thread finished; awaiting finish of 0 more threads
    2020-12-17 13:06:22,076 : INFO : EPOCH - 10 : training on 86715 raw words (56778 effective words) took 0.0s, 1315313 effective words/s
    2020-12-17 13:06:22,077 : INFO : training on a 867150 raw words (568135 effective words) took 0.5s, 1122492 effective words/s
    




    (568135, 867150)




```python
wv = model_W2V.wv
```


```python
wv.most_similar(positive='good')
```

    2020-12-17 13:06:22,089 : INFO : precomputing L2-norms of word weight vectors
    




    [('scan', 0.9812557697296143),
     ('vCards', 0.9642086625099182),
     ('They', 0.9605367183685303),
     ('longer', 0.9592171311378479),
     ('festival', 0.9579607844352722),
     ('behave', 0.9553024768829346),
     ('Simple', 0.9551275968551636),
     ('ipads', 0.9546160697937012),
     ('like', 0.9534077644348145),
     ('dude', 0.9532365798950195)]




```python
wv['help']
```




    array([ 0.26340348, -0.40018958,  0.06040369, -0.19309096, -0.25014228,
            0.11403151,  0.13322787, -0.1535964 ,  0.04666793,  0.22953562,
            0.32465532,  0.15474607, -0.00531764, -0.2743694 , -0.11493958,
            0.11698529,  0.02690452, -0.1391758 , -0.05665627,  0.17294443,
            0.14213072, -0.09188478,  0.00366103,  0.3397651 ,  0.11656451,
            0.17908143, -0.11154556,  0.26916105, -0.11514664, -0.10411742,
           -0.289618  , -0.0189602 ,  0.02211949, -0.16155374, -0.11159767,
           -0.06790099,  0.21793206, -0.11909045,  0.05444815,  0.01792222,
           -0.06722239,  0.02029637,  0.06175743,  0.21486077,  0.31147718,
            0.37647897, -0.30715406,  0.18051036, -0.07471982, -0.14325368,
            0.2517508 ,  0.16290008, -0.23652588,  0.01067481,  0.11774933,
            0.31457067,  0.09200221,  0.07123999,  0.08954549, -0.06245711,
            0.0696683 ,  0.09090552,  0.24627443, -0.09342247, -0.14598   ,
            0.22235683,  0.04680994,  0.11824694, -0.7169352 , -0.26919207,
            0.05049581,  0.00485155, -0.15389952, -0.11464192, -0.14360659,
           -0.22030841,  0.1552505 ,  0.17743054,  0.0372517 ,  0.12613313,
            0.02473311,  0.25920096, -0.10323504, -0.1841724 ,  0.14494286,
            0.15176162,  0.06931927,  0.07450345, -0.25900677, -0.2686803 ,
            0.00593456, -0.10767873, -0.09033866,  0.14090653,  0.02257674,
            0.10058074, -0.13294472,  0.2173957 ,  0.00932731,  0.08059204],
          dtype=float32)




```python
wv.vectors
```




    array([[ 7.5755513e-01,  1.7843774e-02,  3.1630000e-01, ...,
             5.7815367e-01,  5.8852291e-01, -1.6978522e-01],
           [ 7.1542639e-01, -7.6082975e-01,  5.3152555e-01, ...,
             5.2534026e-01, -2.0729542e-02,  3.2103598e-01],
           [ 1.6311506e+00, -1.8997508e+00, -2.5010246e-01, ...,
             2.1062391e-01,  3.3783191e-01, -6.9523197e-01],
           ...,
           [ 4.3708596e-02, -4.8188835e-02,  1.0424881e-02, ...,
             2.1558555e-02,  2.4326772e-03,  4.7256039e-03],
           [-5.6209791e-02, -2.0330665e-03, -8.7956796e-05, ...,
            -5.8318313e-02,  2.5095209e-02, -1.1613940e-02],
           [ 4.5090374e-03, -2.8578965e-02,  1.8759282e-02, ...,
             7.0168213e-03, -2.3256097e-02, -9.8043354e-04]], dtype=float32)




```python
wv.most_similar(positive=['apple','google'], negative = ['man'])
```




    [('sprinkle', 0.7258357405662537),
     ('smartphones', 0.6931923627853394),
     ('Stacks', 0.6855062246322632),
     ('phenomenal', 0.6852372884750366),
     ('mylunch', 0.6822008490562439),
     ('googledoodle', 0.6816586852073669),
     ('technology', 0.6721771955490112),
     ('LatinasInTech\x89Û\x9d', 0.6605688333511353),
     ('Geek', 0.6575497984886169),
     ('nerdheaven', 0.6560829281806946)]




```python

```
