import pandas as pd
import numpy as np
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


class Predictor:
  def __init__(self, algo ):
    if algo == "LR":
      self.filename = "models/logisticMdl.sav"
    elif algo == "DT":
      self.filename = "models/decissionTree.sav"
    elif algo == "naive":
      self.filename = "models/naiveMdl.sav"
    elif algo == "svc":
      self.filename = "models/svcMdl.sav"
    self.mdl = pickle.load(open(self.filename, 'rb'))
    self.cv = pickle.load(open("helpers/countVectorizer.pickel", 'rb'))
    self.le = pickle.load(open("helpers/LabelEncoder.pickel", 'rb'))
  

  def filterNonAscii(self, text):
    split = re.split("\W+", text)
    return " ".join(split)

  def remove_stopwords(self,text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(["br", "href"])
    text=[word for word in text.split(" ") if word not in stopwords]
    return " ".join(text)

  def lemmitize(self,text):
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text.split(" ")])


  def PredictNewsType(self,news):
    if type(news) == str:
      news = [news]
    toBePredicted = pd.DataFrame(news)
    toBePredicted = toBePredicted[0]
    toBePredicted = toBePredicted.apply(lambda x: self.filterNonAscii(x.lower()))
    toBePredicted = toBePredicted.apply(lambda x: self.remove_stopwords(x))
    toBePredicted = toBePredicted.apply(lambda x: self.lemmitize(x))
    outs = self.mdl.predict(self.cv.transform(toBePredicted).toarray())
    outs_prob = self.mdl.predict_proba(self.cv.transform(toBePredicted).toarray())
    ans= []
    for i, out in enumerate(outs):
      ans.append((self.le.inverse_transform([out])[0], np.round(max(outs_prob[i])*100,2)))
    return ans

if __name__ == "__main__":
  p = Predictor("naive")
  print("NEWS CLASSIFIER")
  print("-"*100)
  while 1:
    print()
    print()
    news = input("Enter the news(STOP to end): ")
    if news == "STOP":
      break
    res = p.PredictNewsType(news)
    print()
    print("The NEWS belongs to "+ res[0][0] + " Category with prob : "+ str(res[0][1]))
  print("-"*100)
  print("THANKS FOR USING OUR APP")
