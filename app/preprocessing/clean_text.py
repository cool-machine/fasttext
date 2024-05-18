import re
import unicodedata
import html
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# Load English stop words
stop_words = set(stopwords.words('english'))

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to wordnet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def replace_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text)

def is_valid_utf8(text):
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return False
    else:
        return True

def clean_text(text):
    # Decode HTML entities
    text = html.unescape(text)
    
    # Normalize unicode characters to ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Convert text to lowercase to standardize it
    text = text.lower()
    
    # Remove HTML tags using a regular expression
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs using a regular expression
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = replace_multiple_spaces(text)
    
    # Check for valid UTF-8
    if not is_valid_utf8(text):
        return ""
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    filtered_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tag(tokens)
        if word not in stop_words and len(word) > 1
    ]
    
    # Reconstruct the text without stop words
    return ' '.join(filtered_tokens)

