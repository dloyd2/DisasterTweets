'''
    Daniel Loyd
'''
import re, os
import string
from spellchecker import SpellChecker

import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from DisasterTweets.log import get_logger
LOCATION = os.path.dirname(os.path.abspath(__file__))
log = get_logger('misc')
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
eng_stopwords = set(stopwords.words('english'))

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

def remove_stopwords(text):
    from nltk import word_tokenize
    words = word_tokenize(text)
    new_text = [word for word in words if word not in eng_stopwords]
    return " ".join(new_text)

def clean_tweet(tweet):
    '''
    tweet -> str

    Given a tweet, clean it by removing undesired features.
    Returns the cleaned tweet
    '''
    tweet = tweet.lower()
    tweet = remove_URL(tweet)
    tweet = remove_html(tweet)
    tweet = remove_emoji(tweet)
    tweet = remove_punct(tweet)
    #tweet = correct_spellings(tweet)
    tweet = remove_stopwords(tweet)
    return tweet

def tokenize_tweets(tweets, max_len=30):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    return pad_sequences(sequences, maxlen=max_len, truncating='post', padding='post')

def simplify_tweet(tweet):
    '''
    tweet -> str

    parses the tweet for any extra features
    Returns a list of these features
    '''
    tweet = tweet.lstrip().rstrip()

    # remove the links from the tweet (thanks to )
    tweet = re.sub("http:\S+", "", tweet)
    tweet = re.sub("https:\S+", "", tweet)
    tweet = re.sub(r"[,.;#?!&$:]+\ *", " ", tweet)
    tweet = tweet.replace("  ", " ")
    tweet = tweet.lower()

    tweet = tweet.replace('...', ' ... ').strip()
    tweet = tweet.replace("'", " ' ").strip()

    # Special characters
    tweet = re.sub(r"\x89Û_", "", tweet)
    tweet = re.sub(r"\x89ÛÒ", "", tweet)
    tweet = re.sub(r"\x89ÛÓ", "", tweet)
    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)
    tweet = re.sub(r"\x89ÛÏ", "", tweet)
    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)
    tweet = re.sub(r"let\x89Ûªs", "let's", tweet)
    tweet = re.sub(r"\x89Û÷", "", tweet)
    tweet = re.sub(r"\x89Ûª", "", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"å_", "", tweet)
    tweet = re.sub(r"\x89Û¢", "", tweet)
    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)
    tweet = re.sub(r"åÊ", "", tweet)
    tweet = re.sub(r"åÈ", "", tweet)
    tweet = re.sub(r"JapÌ_n", "Japan", tweet)
    tweet = re.sub(r"Ì©", "e", tweet)
    tweet = re.sub(r"å¨", "", tweet)
    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)

    # Contractions
    tweet = re.sub(r"he's", "he is", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"We're", "We are", tweet)
    tweet = re.sub(r"That's", "That is", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"Can't", "Cannot", tweet)
    tweet = re.sub(r"wasn't", "was not", tweet)
    tweet = re.sub(r"don\x89Ûªt", "do not", tweet)
    tweet = re.sub(r"aren't", "are not", tweet)
    tweet = re.sub(r"isn't", "is not", tweet)
    tweet = re.sub(r"What's", "What is", tweet)
    tweet = re.sub(r"haven't", "have not", tweet)
    tweet = re.sub(r"hasn't", "has not", tweet)
    tweet = re.sub(r"There's", "There is", tweet)
    tweet = re.sub(r"He's", "He is", tweet)
    tweet = re.sub(r"It's", "It is", tweet)
    tweet = re.sub(r"You're", "You are", tweet)
    tweet = re.sub(r"I'M", "I am", tweet)
    tweet = re.sub(r"shouldn't", "should not", tweet)
    tweet = re.sub(r"wouldn't", "would not", tweet)
    tweet = re.sub(r"i'm", "I am", tweet)
    tweet = re.sub(r"I\x89Ûªm", "I am", tweet)
    tweet = re.sub(r"I'm", "I am", tweet)
    tweet = re.sub(r"Isn't", "is not", tweet)
    tweet = re.sub(r"Here's", "Here is", tweet)
    tweet = re.sub(r"you've", "you have", tweet)
    tweet = re.sub(r"you\x89Ûªve", "you have", tweet)
    tweet = re.sub(r"we're", "we are", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"couldn't", "could not", tweet)
    tweet = re.sub(r"we've", "we have", tweet)
    tweet = re.sub(r"it\x89Ûªs", "it is", tweet)
    tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)
    tweet = re.sub(r"It\x89Ûªs", "It is", tweet)
    tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)
    tweet = re.sub(r"who's", "who is", tweet)
    tweet = re.sub(r"I\x89Ûªve", "I have", tweet)
    tweet = re.sub(r"y'all", "you all", tweet)
    tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)
    tweet = re.sub(r"would've", "would have", tweet)
    tweet = re.sub(r"it'll", "it will", tweet)
    tweet = re.sub(r"we'll", "we will", tweet)
    tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)
    tweet = re.sub(r"We've", "We have", tweet)
    tweet = re.sub(r"he'll", "he will", tweet)
    tweet = re.sub(r"Y'all", "You all", tweet)
    tweet = re.sub(r"Weren't", "Were not", tweet)
    tweet = re.sub(r"Didn't", "Did not", tweet)
    tweet = re.sub(r"they'll", "they will", tweet)
    tweet = re.sub(r"they'd", "they would", tweet)
    tweet = re.sub(r"DON'T", "DO NOT", tweet)
    tweet = re.sub(r"That\x89Ûªs", "That is", tweet)
    tweet = re.sub(r"they've", "they have", tweet)
    tweet = re.sub(r"i'd", "I would", tweet)
    tweet = re.sub(r"should've", "should have", tweet)
    tweet = re.sub(r"You\x89Ûªre", "You are", tweet)
    tweet = re.sub(r"where's", "where is", tweet)
    tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)
    tweet = re.sub(r"we'd", "we would", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    tweet = re.sub(r"They're", "They are", tweet)
    tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)
    tweet = re.sub(r"you\x89Ûªll", "you will", tweet)
    tweet = re.sub(r"I\x89Ûªd", "I would", tweet)
    tweet = re.sub(r"let's", "let us", tweet)

    # Character entity references
    tweet = re.sub(r"&gt;", ">", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet)

    # Typos, slang and informal abbreviations
    tweet = re.sub(r"w/e", "whatever", tweet)
    tweet = re.sub(r"w/", "with", tweet)
    tweet = re.sub(r"USAgov", "USA government", tweet)
    tweet = re.sub(r"recentlu", "recently", tweet)
    tweet = re.sub(r"Ph0tos", "Photos", tweet)
    tweet = re.sub(r"exp0sed", "exposed", tweet)
    tweet = re.sub(r"<3", "love", tweet)
    tweet = re.sub(r"amageddon", "armageddon", tweet)
    tweet = re.sub(r"Trfc", "Traffic", tweet)
    tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)
    tweet = re.sub(r"chest/torso", "chest / torso", tweet)
    tweet = re.sub(r"WindStorm", "Wind Storm", tweet)
    tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)
    tweet = re.sub(r"10:38PM", "10:38 PM", tweet)
    tweet = re.sub(r"10:30pm", "10:30 PM", tweet)

    # Separating other punctuations
    tweet = re.sub(r"MH370:", "MH370 :", tweet)
    tweet = re.sub(r"PM:", "Prime Minister :", tweet)
    tweet = re.sub(r"Legionnaires:", "Legionnaires :", tweet)
    tweet = re.sub(r"Latest:", "Latest :", tweet)
    tweet = re.sub(r"Crash:", "Crash :", tweet)
    tweet = re.sub(r"News:", "News :", tweet)
    tweet = re.sub(r"derailment:", "derailment :", tweet)
    tweet = re.sub(r"attack:", "attack :", tweet)
    tweet = re.sub(r"Saipan:", "Saipan :", tweet)
    tweet = re.sub(r"Photo:", "Photo :", tweet)
    tweet = re.sub(r"Funtenna:", "Funtenna :", tweet)
    tweet = re.sub(r"quiz:", "quiz :", tweet)
    tweet = re.sub(r"VIDEO:", "VIDEO :", tweet)
    tweet = re.sub(r"MP:", "MP :", tweet)
    tweet = re.sub(r"UTC2015-08-05", "UTC 2015-08-05", tweet)
    tweet = re.sub(r"California:", "California :", tweet)
    tweet = re.sub(r"horror:", "horror :", tweet)
    tweet = re.sub(r"Past:", "Past :", tweet)
    tweet = re.sub(r"Time2015-08-06", "Time 2015-08-06", tweet)
    tweet = re.sub(r"here:", "here :", tweet)
    tweet = re.sub(r"fires.", "fires .", tweet)
    tweet = re.sub(r"Forest:", "Forest :", tweet)
    tweet = re.sub(r"Cramer:", "Cramer :", tweet)
    tweet = re.sub(r"Chile:", "Chile :", tweet)
    tweet = re.sub(r"link:", "link :", tweet)
    tweet = re.sub(r"crash:", "crash :", tweet)
    tweet = re.sub(r"Video:", "Video :", tweet)
    tweet = re.sub(r"Bestnaijamade:", "bestnaijamade :", tweet)
    tweet = re.sub(r"NWS:", "National Weather Service :", tweet)
    tweet = re.sub(r".caught", ". caught", tweet)
    tweet = re.sub(r"Hobbit:", "Hobbit :", tweet)
    tweet = re.sub(r"2015:", "2015 :", tweet)
    tweet = re.sub(r"post:", "post :", tweet)
    tweet = re.sub(r"BREAKING:", "BREAKING :", tweet)
    tweet = re.sub(r"Island:", "Island :", tweet)
    tweet = re.sub(r"Med:", "Med :", tweet)
    tweet = re.sub(r"97/Georgia", "97 / Georgia", tweet)
    tweet = re.sub(r"Here:", "Here :", tweet)
    tweet = re.sub(r"horror;", "horror ;", tweet)
    tweet = re.sub(r"people;", "people ;", tweet)
    tweet = re.sub(r"refugees;", "refugees ;", tweet)
    tweet = re.sub(r"Genocide;", "Genocide ;", tweet)
    tweet = re.sub(r".POTUS", ". POTUS", tweet)
    tweet = re.sub(r"Collision-No", "Collision - No", tweet)
    tweet = re.sub(r"Rear-", "Rear -", tweet)
    tweet = re.sub(r"Broadway:", "Broadway :", tweet)
    tweet = re.sub(r"Correction:", "Correction :", tweet)
    tweet = re.sub(r"UPDATE:", "UPDATE :", tweet)
    tweet = re.sub(r"Times:", "Times :", tweet)
    tweet = re.sub(r"RT:", "RT :", tweet)
    tweet = re.sub(r"Police:", "Police :", tweet)
    tweet = re.sub(r"Training:", "Training :", tweet)
    tweet = re.sub(r"Hawaii:", "Hawaii :", tweet)
    tweet = re.sub(r"Selfies:", "Selfies :", tweet)
    tweet = re.sub(r"Content:", "Content :", tweet)
    tweet = re.sub(r"101:", "101 :", tweet)
    tweet = re.sub(r"story:", "story :", tweet)
    tweet = re.sub(r"injured:", "injured :", tweet)
    tweet = re.sub(r"poll:", "poll :", tweet)
    tweet = re.sub(r"Guide:", "Guide :", tweet)
    tweet = re.sub(r"Update:", "Update :", tweet)
    tweet = re.sub(r"alarm:", "alarm :", tweet)
    tweet = re.sub(r"floods:", "floods :", tweet)
    tweet = re.sub(r"Flood:", "Flood :", tweet)
    tweet = re.sub(r"MH370;", "MH370 ;", tweet)
    tweet = re.sub(r"life:", "life :", tweet)
    tweet = re.sub(r"crush:", "crush :", tweet)
    tweet = re.sub(r"now:", "now :", tweet)
    tweet = re.sub(r"Vote:", "Vote :", tweet)
    tweet = re.sub(r"Catastrophe.", "Catastrophe .", tweet)
    tweet = re.sub(r"library:", "library :", tweet)
    tweet = re.sub(r"Bush:", "Bush :", tweet)
    tweet = re.sub(r";ACCIDENT", "; ACCIDENT", tweet)
    tweet = re.sub(r"accident:", "accident :", tweet)
    tweet = re.sub(r"Taiwan;", "Taiwan ;", tweet)
    tweet = re.sub(r"Map:", "Map :", tweet)
    tweet = re.sub(r"failure:", "failure :", tweet)
    tweet = re.sub(r"150-Foot", "150 - Foot", tweet)
    tweet = re.sub(r"failure:", "failure :", tweet)
    tweet = re.sub(r"prefer:", "prefer :", tweet)
    tweet = re.sub(r"CNN:", "CNN :", tweet)
    tweet = re.sub(r"Oops:", "Oops :", tweet)
    tweet = re.sub(r"Disco:", "Disco :", tweet)
    tweet = re.sub(r"Disease:", "Disease :", tweet)
    tweet = re.sub(r"Grows:", "Grows :", tweet)
    tweet = re.sub(r"projected:", "projected :", tweet)
    tweet = re.sub(r"Pakistan.", "Pakistan .", tweet)
    tweet = re.sub(r"ministers:", "ministers :", tweet)
    tweet = re.sub(r"Photos:", "Photos :", tweet)
    tweet = re.sub(r"Disease:", "Disease :", tweet)
    tweet = re.sub(r"pres:", "press :", tweet)
    tweet = re.sub(r"winds.", "winds .", tweet)
    tweet = re.sub(r"MPH.", "MPH .", tweet)
    tweet = re.sub(r"PHOTOS:", "PHOTOS :", tweet)
    tweet = re.sub(r"Time2015-08-05", "Time 2015-08-05", tweet)
    tweet = re.sub(r"Denmark:", "Denmark :", tweet)
    tweet = re.sub(r"Articles:", "Articles :", tweet)
    tweet = re.sub(r"Crash:", "Crash :", tweet)
    tweet = re.sub(r"casualties.:", "casualties .:", tweet)
    tweet = re.sub(r"Afghanistan:", "Afghanistan :", tweet)
    tweet = re.sub(r"Day:", "Day :", tweet)
    tweet = re.sub(r"AVERTED:", "AVERTED :", tweet)
    tweet = re.sub(r"sitting:", "sitting :", tweet)
    tweet = re.sub(r"Multiplayer:", "Multiplayer :", tweet)
    tweet = re.sub(r"Kaduna:", "Kaduna :", tweet)
    tweet = re.sub(r"favorite:", "favorite :", tweet)
    tweet = re.sub(r"home:", "home :", tweet)
    tweet = re.sub(r"just:", "just :", tweet)
    tweet = re.sub(r"Collision-1141", "Collision - 1141", tweet)
    tweet = re.sub(r"County:", "County :", tweet)
    tweet = re.sub(r"Duty:", "Duty :", tweet)
    tweet = re.sub(r"page:", "page :", tweet)
    tweet = re.sub(r"Attack:", "Attack :", tweet)
    tweet = re.sub(r"Minecraft:", "Minecraft :", tweet)
    tweet = re.sub(r"wounds;", "wounds ;", tweet)
    tweet = re.sub(r"Shots:", "Shots :", tweet)
    tweet = re.sub(r"shots:", "shots :", tweet)
    tweet = re.sub(r"Gunfire:", "Gunfire :", tweet)
    tweet = re.sub(r"hike:", "hike :", tweet)
    tweet = re.sub(r"Email:", "Email :", tweet)
    tweet = re.sub(r"System:", "System :", tweet)
    tweet = re.sub(r"Radio:", "Radio :", tweet)
    tweet = re.sub(r"King:", "King :", tweet)
    tweet = re.sub(r"upheaval:", "upheaval :", tweet)
    tweet = re.sub(r"tragedy;", "tragedy ;", tweet)
    tweet = re.sub(r"HERE:", "HERE :", tweet)
    tweet = re.sub(r"terrorism:", "terrorism :", tweet)
    tweet = re.sub(r"police:", "police :", tweet)
    tweet = re.sub(r"Mosque:", "Mosque :", tweet)
    tweet = re.sub(r"Rightways:", "Rightways :", tweet)
    tweet = re.sub(r"Brooklyn:", "Brooklyn :", tweet)
    tweet = re.sub(r"Arrived:", "Arrived :", tweet)
    tweet = re.sub(r"Home:", "Home :", tweet)
    tweet = re.sub(r"Earth:", "Earth :", tweet)
    tweet = re.sub(r"three:", "three :", tweet)

    # Hashtags and usernames
    tweet = re.sub(r"IranDeal", "Iran Deal", tweet)
    tweet = re.sub(r"ArianaGrande", "Ariana Grande", tweet)
    tweet = re.sub(r"camilacabello97", "camila cabello", tweet)
    tweet = re.sub(r"RondaRousey", "Ronda Rousey", tweet)
    tweet = re.sub(r"MTVHottest", "MTV Hottest", tweet)
    tweet = re.sub(r"TrapMusic", "Trap Music", tweet)
    tweet = re.sub(r"ProphetMuhammad", "Prophet Muhammad", tweet)
    tweet = re.sub(r"PantherAttack", "Panther Attack", tweet)
    tweet = re.sub(r"StrategicPatience", "Strategic Patience", tweet)
    tweet = re.sub(r"socialnews", "social news", tweet)
    tweet = re.sub(r"NASAHurricane", "NASA Hurricane", tweet)
    tweet = re.sub(r"onlinecommunities", "online communities", tweet)
    tweet = re.sub(r"humanconsumption", "human consumption", tweet)
    tweet = re.sub(r"Typhoon-Devastated", "Typhoon Devastated", tweet)
    tweet = re.sub(r"Meat-Loving", "Meat Loving", tweet)
    tweet = re.sub(r"facialabuse", "facial abuse", tweet)
    tweet = re.sub(r"LakeCounty", "Lake County", tweet)
    tweet = re.sub(r"BeingAuthor", "Being Author", tweet)
    tweet = re.sub(r"withheavenly", "with heavenly", tweet)
    tweet = re.sub(r"thankU", "thank you", tweet)
    tweet = re.sub(r"iTunesMusic", "iTunes Music", tweet)
    tweet = re.sub(r"OffensiveContent", "Offensive Content", tweet)
    tweet = re.sub(r"WorstSummerJob", "Worst Summer Job", tweet)
    tweet = re.sub(r"HarryBeCareful", "Harry Be Careful", tweet)
    tweet = re.sub(r"NASASolarSystem", "NASA Solar System", tweet)
    tweet = re.sub(r"animalrescue", "animal rescue", tweet)
    tweet = re.sub(r"KurtSchlichter", "Kurt Schlichter", tweet)
    tweet = re.sub(r"aRmageddon", "armageddon", tweet)
    tweet = re.sub(r"Throwingknifes", "Throwing knives", tweet)
    tweet = re.sub(r"GodsLove", "God's Love", tweet)
    tweet = re.sub(r"bookboost", "book boost", tweet)
    tweet = re.sub(r"ibooklove", "I book love", tweet)
    tweet = re.sub(r"NestleIndia", "Nestle India", tweet)
    tweet = re.sub(r"realDonaldTrump", "Donald Trump", tweet)
    tweet = re.sub(r"DavidVonderhaar", "David Vonderhaar", tweet)
    tweet = re.sub(r"CecilTheLion", "Cecil The Lion", tweet)
    tweet = re.sub(r"weathernetwork", "weather network", tweet)
    tweet = re.sub(r"withBioterrorism&use", "with Bioterrorism & use", tweet)
    tweet = re.sub(r"Hostage&2", "Hostage & 2", tweet)
    tweet = re.sub(r"GOPDebate", "GOP Debate", tweet)
    tweet = re.sub(r"RickPerry", "Rick Perry", tweet)
    tweet = re.sub(r"frontpage", "front page", tweet)
    tweet = re.sub(r"NewsInTweets", "News In Tweets", tweet)
    tweet = re.sub(r"ViralSpell", "Viral Spell", tweet)
    tweet = re.sub(r"til_now", "until now", tweet)
    tweet = re.sub(r"volcanoinRussia", "volcano in Russia", tweet)
    tweet = re.sub(r"ZippedNews", "Zipped News", tweet)
    tweet = re.sub(r"MicheleBachman", "Michele Bachman", tweet)
    tweet = re.sub(r"53inch", "53 inch", tweet)
    tweet = re.sub(r"KerrickTrial", "Kerrick Trial", tweet)
    tweet = re.sub(r"abstorm", "Alberta Storm", tweet)
    tweet = re.sub(r"Beyhive", "Beyonce hive", tweet)
    tweet = re.sub(r"IDFire", "Idaho Fire", tweet)
    tweet = re.sub(r"DETECTADO", "Detected", tweet)
    tweet = re.sub(r"RockyFire", "Rocky Fire", tweet)
    tweet = re.sub(r"Listen/Buy", "Listen / Buy", tweet)
    tweet = re.sub(r"NickCannon", "Nick Cannon", tweet)
    tweet = re.sub(r"FaroeIslands", "Faroe Islands", tweet)
    tweet = re.sub(r"yycstorm", "Calgary Storm", tweet)
    tweet = re.sub(r"IDPs:", "Internally Displaced People :", tweet)
    tweet = re.sub(r"ArtistsUnited", "Artists United", tweet)
    tweet = re.sub(r"ClaytonBryant", "Clayton Bryant", tweet)
    tweet = re.sub(r"jimmyfallon", "jimmy fallon", tweet)

    words = [x.lstrip().rstrip() for x in tweet.split()]

    # remove all the stop words
    words = [x for x in words if x and x not in eng_stopwords]

    # check if tweet is not empty
    if not words:
        return ""

    # replace words with their stem words: https://www.nltk.org/howto/stem.html
    # words = [stemmer.stem(x) for x in words]

    # replace @usermentions with simply user
    words = ['user' if x.startswith('@') else x for x in words]

    tweet = " ".join(words)
    tweet = re.sub(r"[,.;@#?!&$:]+\ *", " ", tweet)

    # now we can return the valid tweet
    return tweet
