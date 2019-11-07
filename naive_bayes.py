import pandas as pd
import numpy as np
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split

# Loading the csv file and dropping columns
data = pd.read_csv('train.csv', encoding = "ISO-8859-1")
pd.options.display.max_colwidth = 1000
to_drop = ['id',
           	'title',
           	'author',]
data.drop(to_drop, inplace=True, axis=1)
data.text = data.text.astype(str)
dataText = data['text']
dataLabels = data['label']

# Text Preprocessing and removing stop words
def preprocess (text):
    stop_words = nltk.corpus.stopwords.words('english')
    custom_words = ['the','would','said','one','also','like','we', 'it', 'new','could',
                    'he','going','another','need','right','much','we','ve','say','made','she','may']
    stop_words.extend(custom_words)
    
    train = text.lower()
    train = train.replace('[^a-z\s]+',' ')
    train = train.replace('[^\w\s] ', ' ')
    train = train.replace('(\s+)',' ')
    train = train.split()
  
    filtered_sentence = [w for w in train if not w in stop_words] 
    filtered_sentence = [] 
  
    for w in train: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    train = filtered_sentence
    return train

class TextClass:
    def __init__(self, labels):
        self.classes = labels
        
    def bagOfWords(self, text, bowIndex):
        if isinstance(text, np.ndarray): text = text[0]
        if text is not None:
            for token in text.split():
                self.bow[bowIndex][token] += 1
    
    def train(self, data, labels):
        self.data = data
        self.labels = labels
        self.bow = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
            
        # Creating Bag of Words for each label
        for labelIndex, label in enumerate(self.classes):
            labelsData = self.data[self.labels == label]
            processed_data = [preprocess(labelText) for labelText in labelsData]
            processed_data = pd.DataFrame(data = processed_data)
            np.apply_along_axis(self.bagOfWords, 1, processed_data, labelIndex)
      
        prob_classes = np.empty(self.classes.shape[0])
        wordsCount = []
        labelWcount = np.empty(self.classes.shape[0])
        
        # Calculating prior probability of each label and word count for each label
        for labelIndex, label in enumerate(self.classes):
            prob_classes[labelIndex] = np.sum(self.labels == label) / float(self.labels.shape[0])
            probL = prob_classes[labelIndex]
            count = list(self.bow[labelIndex].values())
            self.bowL = self.bow[labelIndex]
            labelWcount[labelIndex] = np.sum(np.array(list(self.bowL.values()))) + 1                            
            wordsCount += self.bowL.keys()
        # Vocabulary
        self.vocabulary = np.unique(np.array(wordsCount))
                                  
        # Denominator value                                     
        denoms = np.array([labelWcount[labelIndex] + self.vocabulary.shape[0] + 1 for labelIndex, label in enumerate(self.classes)])
        
        self.labelsInfo = [(self.bowL, probL, denoms[labelIndex]) for labelIndex, label in enumerate(self.classes)]                               
        self.labelsInfo = np.array(self.labelsInfo)                                 
    
    # Finds the Posterior Probability
    def testProbability(self, testText):                                                              
        # Likelihood Probability
        likelihood = np.zeros(self.classes.shape[0])
        for labelIndex, label in enumerate(self.classes): 
            for word in testText:                                               
                tokenCounts = self.labelsInfo[labelIndex][0].get(word, 0) + 1                           
                wordProb = tokenCounts / float(self.labelsInfo[labelIndex][2])                              
                likelihood[labelIndex] += np.log(wordProb)
        # Posterior Probability
        posterior = np.empty(self.classes.shape[0])
        for labelIndex, label in enumerate(self.classes):
            posterior[labelIndex] = likelihood[labelIndex] + np.log(self.labelsInfo[labelIndex][1])
        return posterior
    
    # Testing and evaluation Functions
    def test(self, testData):
        pred = []
        for text in testData:                               
            processed_testData = preprocess(text)                            
            posterior = self.testProbability(processed_testData)
            pred.append(self.classes[np.argmax(posterior)])
        return np.array(pred)
    
    def evaluate(self, testData):
        pred = []
        for text in testData:                                
            processed_testData = preprocess(text)                              
            posterior = self.testProbability(processed_testData)
            pred.append(self.classes[np.argmax(posterior)])
        print(pred)
        return np.array(pred)

trainData, testData, trainLabels, testLabels = train_test_split(dataText, dataLabels, shuffle=True, test_size=0.30, random_state=42, stratify = dataLabels)
classes=np.unique(trainLabels)

textClass = TextClass(classes)
print ('Training....')
textClass.train(trainData, trainLabels)
print ('Training Completed')

testPrediction = textClass.test(testData)
test_acc = np.sum(testPrediction == testLabels) / float(testLabels.shape[0])
print ("Test count: ",testLabels.shape[0])
print ("Test accuracy: ",test_acc)

#testString = "Donald Trump has announced that the US will recognize Israel’s sovereignty over the Golan Heights, captured from Syria in 1967, in a dramatic move likely to bolster Benjamin Netanyahu’s hopes to win re-election, but which will also provoke international opposition. Previous US administrations have treated Golan Heights as occupied Syrian territory, in line with UN security council resolutions. Trump declared his break with that policy, in a tweet on Thursday. He said: After 52 years it is time for the United States to fully recognize Israel’s sovereignty over the Golan Heights, which is of critical strategic and security importance to the State of Israel and Regional Stability!' By defying a 52-year-old unanimously adopted UN resolution on “inadmissibility of the acquisition of territory by war”, Trump has also broken the postwar norm of refusing to recognise the forcible annexation of territory – which has underpinned western and international opposition to the Russian annexation of Crimea. “The United States relies on these core principles regarding peaceful dispute resolution and rejecting acquisition of territory by force,” Tamara Cofman Wittes, the former deputy assistant secretary of state for Near Eastern affairs, wrote on Twitter. Wittes, now a senior fellow at the Brookings Institution, added the move “yanks the rug out from under US policy opposing Russia’s annexation of Crimea, as well as US views on other disputed territories”. Netanyahu, the Israeli PM, quickly tweeted his gratitude for Trump’s gesture. “At a time when Iran seeks to use Syria as a platform to destroy Israel, President Trump boldly recognizes Israeli sovereignty over the Golan Heights,” the Israeli prime minister wrote. “Thank you President Trump!” The announcement came as Netanyahu was hosting the US secretary of state, Mike Pompeo, in Jerusalem. “President Trump has just made history,” Netanyahu said. “I called him. I thanked him on behalf of the people of Israel. The message that President Trump has given the world is that America stands by Israel.” Pompeo said: “President Trump tonight made the decision to recognise that hard-fought real estate, that important place is proper to be sovereign part of the state of Israel.” He added: “The people of Israel should know that the battles they fought, the lives that they lost on that very ground, were worthy, meaningful and important for all time.” The announcement marks a diplomatic coup for Netanyahu, less than three weeks before a close fought election, and four days before he is due to visit Washington. Trump denied his announcement was intended to help Netanyahu hold on to office, even suggesting he had been unaware the election was imminent. I wouldn’t even know about that. I have no idea, Trump told Fox News. He said he had been thinking about recognising the Israeli annexation for a long time. This is sovereignty, this is security, this is about regional security, he said."
#predLabels = nb.evaluate(testString)
