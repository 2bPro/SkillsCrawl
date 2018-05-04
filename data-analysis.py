# Import the necessary libraries and tools
# The analysis of the data has been done with the help of
## the libraries docs, Stackoverflow and the article found at
### goo.gl/LYLCv5
import matplotlib.pyplot as plt
import pylab, re, warnings, lda, logging
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans

from wordcloud import WordCloud

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file

from string import punctuation
from collections import Counter

# Function that returns the tokens extracted from a job's description
# It takes as input parameter the description attribute of a job object
def tokenizer(text):
    # Clean symbols
    job_text = re.sub("[^a-zA-Z.+]", "  ", text)
    # Lowercase and split into words
    job_tokens_init = job_text.lower().split()
    # Filter out stop words
    stop_words = stopwords.words("english")
    # Add additional stop words depending on the findings
    stop_words.append("k")
    stop_words.append("new")
    stop_words.append("you")
    stop_words.append("we")
    stop_words.append("job")
    stop_words.append("u")
    stop_words.append("it'")
    stop_words.append("'s")
    stop_words.append("n't")
    stop_words.append("mr.")
    stop_words.append("day")
    stop_words.append("days")
    stop_words.append("ago")
    stop_words.append("apply")
    stop_words.append("company")
    stop_words.append("cooky")
    stop_words.append("edinburgh")
    stop_words.append("glasgow")
    stop_words.append("aberdeen")
    stop_words.append("scotland")
    stop_words.append("chess")
    stop_words.append("ovarian")
    stop_words.append("cancer")
    stop_words.append("patient")
    stop_words.append("medica")
    stop_words.append("boyall")
    stop_words.append("chemist")
    stop_words.append("medicine")
    stop_words.append("jpmorgan")
    stop_words.append("morgan")
    stop_words.append("chase")
    stop_words.append("aged")
    stop_words.append("staff")
    stop_words.append("training")
    stop_words.append("humanity")
    stop_words.append("william")
    stop_words.append("grant")
    stop_words.append("civil")
    stop_words.append("uk")
    stop_words.append(".")
    stop_words.append("+")

    stop_words = set(stop_words)
    
    # Extract stop words from tokens
    job_tokens = [word for word in job_tokens_init if word not in stop_words]

    # Create an NLTK lemmatizer 
    lemmatizer = WordNetLemmatizer()
    # Apply lemmatization on the tokens
    job_tokens = [lemmatizer.lemmatize(word) for word in job_tokens]

    # Create an NLTK stemmer
    #stemmer = PorterStemmer()
    # Apply stemming on the tokens
    #job_tokens = [stemmer.stem(word) for word in job_tokens]
    
    # Encode tokens
    tokens = [word.encode('utf-8', 'strict') for word in job_tokens]
    # Calculate the percentage of tokens extracted from the description
    extraction = (len(tokens)*100)/len(job_tokens_init)

    #return extraction
    return tokens
# End of the "tokenizer()" function

# --------------------------------------------------------------------------- #

# Function that saves the tokens in the dataframe under the "tokens" column     
def tokenize():
    # Print the progress
    print('---')
    print('Tokenizing text and adding tokens to dataframe...')
    # Create the column and populate it with the returns of the "tokenizer()" function
    data['tokens']= data['descr'].map(tokenizer)
    # Print the data shape of the dataframe (rows x columns)
    print 'data shape: ', data.shape
# End of the "tokenize()" function

# Function that calculates the average percentage of tokens extracted from the descriptions
def getTokenAverage():
    # Initialise the sum of the jobs percentages
    count_perc = 0;
    # Initialise the number of jobs
    count_jobs = 0;
    # For every description in the dataframe
    for job in data['descr']:
        # Increment the number of jobs
        count_jobs += 1
        # Add the percentages together
        count_perc += tokenizer(job)
    
    # Calculate the average percentage
    avg = count_perc/count_jobs
    # Print the average
    print('Average tokenization: ' + str(avg))
# End of the "getTokenAverage()" function

# --------------------------------------------------------------------------- #

# Function that manages job advertisement duplicates
def dropDuplicates():
    # Print progress
    print('---')
    print('Dropping duplicates...')
    # Delete duplicates based on whether they share the same description and 
    ## the same date at the same time
    data.drop_duplicates(subset=['date', 'descr'], keep="first", inplace=True)
    # Print the data shape of the dataframe (rows x columns)
    print 'data shape: ', data.shape
    # Save the data to an external ".csv" file
    data.to_csv("assets/corpus/corpus-noDup.csv")
# End of the "dropDuplicates()" function

# --------------------------------------------------------------------------- #

# Function that creates and displays a popularity bar chart
# It takes as input parameters x = the number of bars and y = the measurement units
def popularityGraph(x, y):
    # Print progress
    print('---')
    print('Showing category popularity graph...')
    # Set bar colors
    colors = list(['#ab1c57', '#e82563', '#f38eaf', '#9474cc', '#673db6', 
                   '#4b1b8b', '#1749a0', '#7885ca', '#2995f3', '#81d3fa',
                   '#4fcfe0', '#4fb4ab', '#0d8377', '#086064', '#459e49', 
                   '#adb32e', '#ccdb3b', '#ffed58', '#fabf2f'
                  ])
    # Create and display the plot with the use of matplotlib
    data.category.value_counts().plot(kind='bar', grid=True, figsize=(x, y), color=colors)
# End of the "popularityGraph()" function

# --------------------------------------------------------------------------- #

# Function that prints out the top ten tokens per job depending on their frequency distribution
# It takes as input parameter the number of tokens to be returned per job
def topTokensPerJob(no):
    # Print progress
    print('---')
    print('Showing tokens per job...')
    # For every job and list of tokens in the dataframes columns
    # Select only the first ten jobs
    for job, tokens in zip(data['title'].head(10), data['tokens'].head(10)):
        # Print the title of the job
        print('job title:', job)
        # Print the tokens by applying NLTK frequency distribution
        print('tokens:', FreqDist(tokens).most_common(no))
        print('---')
# End of the "topTokensPerJob()" function

# --------------------------------------------------------------------------- #

# Function that returns the top tokens per category
# It takes as input parameter the job category and the number of tokens to return
def keywords(category, top):
    # Extract the tokens for the specified category
    tokens = data[data['category'] == category]['tokens']
    # Prepare a list to hold all the tokens of the category
    allTokens = []
    # For every list of tokens extracted
    for token_list in tokens:
        # Spill them in the "allTokens" list
        allTokens += token_list
    # Apply NLTK frequency distribution on the list of tokens
    counter = FreqDist(allTokens)
    # Return top tokens
    return counter.most_common(top)
# End of the "keywords" function

# --------------------------------------------------------------------------- #

# Function that makes use of the "keywords" function to return all the categories
## and their respective top tokens
def topTokensPerCat():
    # Print progress  
    print('---')
    print('Showing top tokens per category...')
    # For every category in the dataframe's "category" column
    for category in set(data['category']):
        # Print the category
        print('category:', category)
        # Print the top tokens
        print('top keywords:', keywords(category))
        print('---')
# End of the "topTokensPerCat()" function

# --------------------------------------------------------------------------- #

# Function that creates a TF-IDF matrix
def tfidfMatrix():
    # Print progress
    print('---')
    print('Creating a tfidf matrix...')
    # Create a vectorizer with the help of sklearn
    vectorizer = TfidfVectorizer(
                # Set the minimum number of descriptions for a term's occurence
		min_df=5,
                # Analyse full words
		analyzer='word',
                # Use engish stop words
                stop_words='english',
                # Set the size of word sets to be taken in consideration
		ngram_range=(1, 2)
    )

    # Use the tokens in the dataframe's "tokens" column as input to the vectorizer
    vz = vectorizer.fit_transform(list(data['tokens'].map(lambda tokens: '  '.join(tokens))))
    # Print the data shape of the dataframe (rows x columns)
    print 'data shape: ', vz.shape
    # Create a tfidf dataframe containing the tokens and their respective tfidf scores
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
    tfidf.columns = ['tfidf']
    # Save the matrix in a ".csv" file
    tfidf.to_csv('assets/dataframes/tfidf.csv')
    # Return the vectorizer and dataframe for further use
    return vectorizer, vz, tfidf
# End of the "tfidfMatrix()" function

# --------------------------------------------------------------------------- #

# Function that creates and displays word clouds
# It takes as input parameter the terms to be taken in consideration
def wordCloud(terms):
    # Join all the terms together
    text = terms.index
    text = '  '.join(list(text))
    # Create a word cloud with the use of the wordcloud library
    wordcloud = WordCloud(max_font_size=40).generate(text)
    # Set the figure size
    plt.figure(figsize=(25, 25))
    plt.imshow(wordcloud, interpolation="bilinear")
    # Turn the visibility of axis off
    plt.axis("off")
    # Show the figure
    plt.show()
# End the "wordCloud()" function

# --------------------------------------------------------------------------- #

# Function that creates a TF-IDF histogram 
# It takes as input parameters the tfidf dataframe and the figure size
def tfidfHistogram(tfidf, x, y):
    # Print progress
    print('---')
    print('Showing a tfidf histogram...')
    # Create histogram
    tfidf.tfidf.hist(color="purple", bins=50, figsize=(x,y))
# End of the "tfidfHistogram" function

# -------------------------------------------------------------------------- #

# Function that prints the tokens with the top lowest TF-IDF scores
# It takes as input parameters the tfidf dataframe and the number of tokens to return
def topLowestTfidf(tfidf, top):
    # Print progress
    print('---')
    print('Showing top {} words with lowest tfidf scores...'.format(top))
    # Sort the values in ascending order by score and select the first top tokens
    print tfidf.sort_values(by=['tfidf'], ascending=True).head(top)
# End of the "topLowestTfidf()" function

# -------------------------------------------------------------------------- #

# Function that prints the tokens with the top highest TF-IDF scores
# It takes as input parameters the tfidf dataframe and the number of tokens to return
def topHighestTfidf(tfidf, top):
    # Print progress
    print('---')
    print('Showing top {} words with highest tfidf scores...'.format(top))
    # Sort the values in descending order by score and select the first top tokens
    print tfidf.sort_values(by=['tfidf'], ascending=False).head(top)
# End of the "topHighestTfidf()" function

# -------------------------------------------------------------------------- #

# Function that applies SVD to reduce tfidf dataframe dimensions to 50
# It takes as input parameter the vectorizer created during the application of TF-IDF
def applySVD(vz):
    # Print progress
    print('---')
    print('Reducing tfidf matrix dimensions to 50 using SVD...')
    # Apply SVD with the help of sklearn
    svd = TruncatedSVD(n_components=50, random_state=0)
    # Create a model with the results
    svd_tfidf = svd.fit_transform(vz)
    # Print the data shape of the model (rows x columns)
    print 'data shape: ', svd_tfidf.shape
    # Return the svd model for further use
    return svd_tfidf
# End of the "applySVD()" function

# ------------------------------------------------------------------------- #

# Function that applies t-SNE to reduce svd model dimensions to two
# It takes as input parameter the svd model
def redDimension(svd_tfidf):
    # Print progress
    print('---')
    print('Reducing tfidf matrix dimensions to two using TSNE...')
    # Apply t-SNE with the help of sklearn
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    # Create a model with the results
    tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
    # Print the data shape of the dataframe (rows x columns)
    print 'data shape: ', tsne_tfidf.shape
    
    # Create a dataframe out of the tsne model
    tsne_tfidf_df = pd.DataFrame(tsne_tfidf)
    # Set its columns
    tsne_tfidf_df.columns = ['x', 'y']
    tsne_tfidf_df['category'] = data['category']
    tsne_tfidf_df['description'] = data['descr']
    # Return the tsne dataframe for further use
    return tsne_tfidf_df
# End of the "redDimension()" function

# ------------------------------------------------------------------------ #

# Function that plots the tsne dataframe on a scatter plot
# It takes as input parameter the tsne dataframe
def groupTfidf(tsne_tfidf):
    # Group the tokens by the job category
    groups = tsne_tfidf.groupby('category')
    # Set the size of the plot and the margins
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.margins(0.05)
    # For every token and group in groups
    for name, group in groups:
        # Plot the tokens
        ax.plot(group.x, group.y, marker='o', linestyle='', label=name)
    # Add a legend specifying the categories
    ax.legend()
    # Show plot
    plt.show()
# End of the "groupTfidf()" function

# ------------------------------------------------------------------------ #

# Function that applies K-means based on the tfidf vectorizer
# This function also plots the K-means results on a scatter plot with the use of Bokeh
# It takes as input parameters the tfidf vectorizer and the number of clusters 
def applyKMeans(vz, clusterNo):
    # Print progress
    print('---')
    print('Applying k-means clustering...')
    # Ignore any deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Save the interactive plot as an html file
    output_file('assets/chartsAndPlots/interactivePlots/kmeans.html')
    # Set the colormap
    colormap = np.array(["#e33e39", "#e62562", "#982dad", "#6a0080", "#653eb5", 
                         "#4253b3", "#2998f2", "#82d4fa", "#50cdde", "#2ca397", 
                         "#449c47", "#ccd93b", "#ffec59", "#ff8c00", "#fc5826",
                         "#8a6c62", "#616161", "#768d99", "#491c8a", "#ab1d56",
                         "#7d0028", "#18499e", "#086063", "#80741d", "#f28fae",
                         "#9374cc", "#66479e", "#ff8a66", "#7986c9", "#aeb32e",
                         "#fabd2f", "#5c413a", "#045a80", "#753700"
                        ])

    # Set the figure features
    plot_kmeans = bp.figure(
            plot_width=700, 
            plot_height=600, 
            title="KMeans clustering of job advertisements in Scotland",
            tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
            x_axis_type=None,
            y_axis_type=None,
            min_border=1
    )
    
    # Create a k-means model with the help of sklearn 
    kmeans_model = MiniBatchKMeans(
            n_clusters=clusterNo, 
            init='k-means++',
            n_init=1,
            init_size=1000,
            batch_size=1000,
            verbose=False,
            max_iter=1000
    )

    # Import the tfidf vectorizer into the k-means model
    kmeans = kmeans_model.fit(vz)
    # Create the clusters
    kmeans_clusters = kmeans.predict(vz)
    # Calculate distances
    kmeans_distances = kmeans.transform(vz)
    
    # Use t-SNE to reduce dimensions
    tsne_model  = TSNE(n_components=2, verbose=1, random_state=0)
    # Apply distances to the two-dimmensional model
    tsne_kmeans = tsne_model.fit_transform(kmeans_distances)

    # Create a dataframe with the results
    kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
    # Set the columns of the dataframe
    kmeans_df['cluster'] = kmeans_clusters
    kmeans_df['category'] = data['category']
    kmeans_df['title'] = data['title']
    kmeans_df['description'] = data['descr']
    kmeans_df['colors'] = colormap[kmeans_clusters]

    # Save k-means dataframe to an external ".csv" file
    kmeans_df.to_csv("assets/dataframes/kmeans.csv")
    # Print progress
    print('Plotting kmeans results...')
    
    # Plot the dataframe
    plot_kmeans.scatter(
            x='x', y='y',
            color='colors',
            source=kmeans_df)

    # Set plot tools
    hover = plot_kmeans.select(dict(type=HoverTool))
    hover.tooltips={"cluster": "@cluster", "category": "@category", "job title": "@title"}
    # Show plot in browser
    show(plot_kmeans)
# End of the "applyKmeans()" function

# --------------------------------------------------------------------------- #

# Function that applies and plots LDA based on the job descriptions
# It takes as input parameter the number of topics to focus on
def applyLDA(topicsNo):
    # Managing warnings and errors
    logging.getLogger("lda").setLevel(logging.WARNING)
    # Save the interactive plot as an html file
    output_file('assets/chartsAndPlots/interactivePlots/lda.html')
    # Create a vectorizer with the use of sklearn 
    cvectorizer = CountVectorizer(min_df=4, max_features=10000, tokenizer=tokenizer, ngram_range=(1,2))
    # Use the descriptions in the dataframe's "descr" column as input to the vectorizer
    cvz = cvectorizer.fit_transform(data['descr'])
    # Create an LDA model
    lda_model = lda.LDA(n_topics=topicsNo, n_iter=500)
    # Create topics based on that model
    X_topics = lda_model.fit_transform(cvz)
    # Words to take in consideration to be considered similar
    n_top_words = 8
    # List to hold the topic summaries
    topic_summaries = []
    # Create a vocabulary vectorizer
    topic_word = lda_model.topic_word_
    vocab = cvectorizer.get_feature_names()
    # Apply t-SNE to reduce dimensions
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_lda = tsne_model.fit_transform(X_topics)
    # Create a list of topics
    doc_topic = lda_model.doc_topic_
    lda_keys = []
    # For every description in the dataframe
    for i, t in enumerate(data['descr']):
        # Create topic
        lda_keys += [doc_topic[i].argmax()]
        # Set a colormap for the plot
	colormap = np.array(["#e33e39", "#e62562", "#982dad", "#6a0080", "#653eb5", 
                             "#4253b3", "#2998f2", "#82d4fa", "#50cdde", "#2ca397", 
                             "#449c47", "#ccd93b", "#ffec59", "#ff8c00", "#fc5826",
                             "#8a6c62", "#616161", "#768d99", "#491c8a", "#ab1d56",
                             "#7d0028", "#18499e", "#086063", "#80741d", "#f28fae",
                             "#9374cc", "#66479e", "#ff8a66", "#7986c9", "#aeb32e",
                             "#fabd2f", "#5c413a", "#045a80", "#753700"
                            ])
    # Set scatter plot features
    plot_lda = bp.figure(
            plot_width=800, 
            plot_height=700, 
            title="LDA topic visualization of job descriptions",
            tools="pan,wheel_zoom, box_zoom, reset, hover, previewsave",
            x_axis_type=None, 
            y_axis_type=None, 
            min_border=1
    )

    # Create a dataframe with the results
    lda_df = pd.DataFrame(tsne_lda, columns=['x', 'y'])
    # Set the columns of the dataframe
    lda_df['category'] = data['category']
    lda_df['title'] = data['title']
    lda_df['description'] = data['descr']
    lda_df['colors'] = colormap[lda_keys]

    lda_df['topic'] = lda_keys
    lda_df['topic'] = lda_df['topic'].map(str)

    # Print progress
    print('Plotting LDA results...')
    # Plot the dataframe
    plot_lda.scatter(
            x='x', y='y',
            color='colors',
            source=lda_df, 
            legend='category'
    )
    # Set plot tools
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips={"topic": "@topic", "category": "@category", "job title": "@title"}
    # Show in browser 
    show(plot_lda)
# End of the "applyLDA()" function

# ------------------------------------------------------------------------ #

# Function that saves soft skills that match with individual descriptions in a dataframe column
def getSoftSkills():
    # Function that matches the soft skills with the tokens of individual descriptions
    # It takes as input parameter the tokens of a description
    def getSkills(tokens):
        # Set a list of soft skills keywords
        skills = ["communication", "bilingual", "writing", "listening", 
                  "speaking", "presentation", "teamwork", "strategic",
                  "coaching", "mentoring", "delegating", "diplomatic", 
                  "managing", "supervising", "inspired", "persuasive", 
                  "motivating", "persevering", "perfectionist", "collaborative", 
                  "confident", "adaptable", "resilient", "assertive", 
                  "competitive", "leader", "friendly", "enthusiastic",                            
                  "empathy", "solver", "critical", "innovative", 
                  "troubleshooting", "artistic", "organized", 
                  "punctual", "aware", "competent", "entrepreneurial", 
                  "analytic","learning", "independent", "team-player"
                 ]
        
        # Create a list to hold the matching skills
        matches = []
        # Apply frequency distribution on the list of tokens
        freqDist = FreqDist(tokens)
        # For every skill in the list of skills
        for s in skills:
            # If there is a match between one of the tokens and the skill
            if freqDist[s] > 0:
                # Add the skill to the "matches" list
                matches.append(s)
        # Return the list of matched skills
        return matches
    # End of the "getSkills()" function

    # ------------------------------------ #

    # Print progress
    print('---')
    print('Searching for soft skills and saving them to the dataframe...')
    # Create a "soft-skills" column in the dataframe
    # Populate it with the list of skills that match the descriptions
    data['soft-skills'] = data['tokens'].map(getSkills)
    # Print the data shape of the dataframe (rows x columns)
    print "data shape: ", data.shape
# End of the "getSoftSkills()" function

# ------------------------------------------------------------------------ #

# Function that returns the top most common soft skills per specified category
# It takes as input parameters the category and the number of skills to return
def softSkills(category, top):
    # Extract the lists of soft skills for that specific category
    skills = data[data['category'] == category]['soft-skills']
    # Create a list to hold all the skill keywords
    allSkills = []
    # For every skill list in the extracted skills
    for skill_list in skills:
        # Spill the contents of the list in the "allSkills" list
        allSkills += skill_list
    # Apply frequency distribution on the skills list
    counter = FreqDist(allSkills)
    # Select the top skills
    return counter.most_common(top)
# End of the "softSkills()" function

# ------------------------------------------------------------------------ #

# Function that prints all the categories and their respective top soft skills
# It takes as input parameter the number of skills to be returned per category
def topSoftSkillsPerCat(top):
    # Print progress    
    print('---')
    print('Showing top soft skills per category...')
    # For every category in the dataframe's "category" column 
    for category in set(data['category']):
        # Print the category
        print('category:', category)
        # Print the top soft skills for that category 
        print('top soft skills:', softSkills(category, top))
        print('---')
# End of the "topSoftSkillsPerCat()" function

# ------------------------------------------------------------------------ #

# Function that saves hard skills that match with individual descriptions in a dataframe column
def getHardSkills():
    # Function that matches the hard skills with the tokens of individual descriptions
    # It takes as parameter the tokens of a description
    def getSkills(tokens):
        # Set a list of hard skill keywords
        skills = ['java', 'c', 'c++', 'c#', 'python', '.net', 'visual basic', 
                  'php', 'javascript', 'pascal', 'swift', 'perl', 'ruby',
                  'assembly', 'r', 'objective-c', 'go', 'matlab', 'sql',
                  'scratch', 'angular', 'mean', 'hadoop', "jquery", "arduino",
                  'nosql', 'node', 'mongodb','scala', "microsoft", "linux", 
                  'shell', 'typescript', 'css', 'html', 'saas', 'webgl', 
                  'react', 'express', 'raspberrypi', 'photoshop'
                 ]
        
        # Create a list to hold the matching skills
        matches = []
        # Apply frequency distribution on the list of tokens
        freqDist = FreqDist(tokens)
        # For every skill in the skills list
        for s in skills:
            # If there is a match between one of the tokens and the skill
            if freqDist[s] > 0:
                # Add the skill to the matches list
                matches.append(s)
        # Return the list of matches
        return matches
    # End of the "getSkills()" function

    # ----------------------------------------- #

    # Print progress
    print('---')
    print('Searching for hard skills and saving them to the dataframe...')
    # Create a "hard-skills" column in the dataframe
    # Poulate it with the list of skills that match the descriptions
    data['hard-skills'] = data['tokens'].map(getSkills)
    # Print the data shape of the dataframe (rows x columns)
    print "data shape: ", data.shape
# End of the "getHardSkills()" function

# -------------------------------------------------------------------------- #

# Function that returns the top most common hard skills per specified category
# It takes as input parameters the category and the number of skills to return 
def hardSkills(category, top):
    # Extract the lists of hard skills for that specific category
    skills = data[data['category'] == category]['hard-skills']
    # Create a list to hold all the skill keywords
    allSkills = []
    # For every skill list in th extracted skills
    for skill_list in skills:
        # Spill the contents of the list in the "allSkills" list
        allSkills += skill_list
    # Apply frequency distribution on the skills list
    counter = FreqDist(allSkills)
    # Select the top skills
    return counter.most_common(top)
# End of the "hardSkills()" function

# ------------------------------------------------------------------------ #

# Function that prints all the categories and their respective hard skills
# It takes as input parameter the number of skills to be returned per category
def topHardSkillsPerCat():
    # Print progress    
    print('---')
    print('Showing top hard skills per category...')
    # For every category in the dataframe's "category" column
    for category in set(data['category']):
        # Print the category
        print('category:', category)
        # Print the top hard skills for that category
        print('top hard skills:', hardSkills(category, top))
        print('---')
# End of the "topHardSkillsPerCat()" function

# ------------------------------------------------------------------------ #

# Function that acts as module for plotting skills
# It takes as input parameters the skills and the title of the figure
def plotSkills(skills, title):    
    # Create a dataframe out of the skills and their frequency
    skills_frame = pd.DataFrame(skills.items(), columns=['Term', 'Frequency'])
    # Sort the skills by frequency
    skills_frame.sort_values(by='Frequency', ascending=False, inplace=True)
    # Prepare plot
    skills_plot = skills_frame.plot(
            x='Term', 
            kind="bar", 
            legend=None,
            title=title,
            color='purple'
    )
    # Set plot label
    skills_plot.set_ylabel('Percentage Appearing in Job Ads')
    # Create figure
    fig = skills_plot.get_figure()
    # Return figure and skills dataframe
    return fig, skills_frame
# End of "plotSkills()" function

# ----------------------------------------------------------------------- #

# Function that returns a plot of soft skills depending on their frequency
def overallSoftSkills():
    # Create a list to hold all skills
    allSkills = []
    # For every list of skills in the dataframe's "soft-skills" column
    for skill_list in data['soft-skills']:
        # Spill the contents of the list in the "allSkills" list
        allSkills += skill_list
    
    # Apply frequency distribution on the skills list
    counter = FreqDist(allSkills)
    # Plot the list of skills depending on their frequencies 
    ## using the "plotSkills()" function
    print plotSkills(dict(counter), 'Overall Soft Skill Demand in Scotland')
# End of the "overallSoftSkills()" function

# ---------------------------------------------------------------------- #

# Function that creates a plot of soft skills of a specific category
# This function also returns the skills and their respective frequencies
def softSkillsPerCat(category):
    # Extract the lists of soft skills for that specific category
    skills = data[data['category'] == category]['soft-skills']
    # Create a list to hold all the skill keywords
    allSkills = []
    # For every skill list in the extracted skills
    for skill_list in skills:
        # Spill the contents of the list in the "allSkills" list
        allSkills += skill_list
    # Apply frequency distribution on the skills list
    counter = FreqDist(allSkills)
    # Plot the list of skills depending on their frequencies
    ## using the "plotSkills()" function
    print plotSkills(dict(counter), "Soft Skill Demand in Scotland for {}".format(category))
    # Return a dictionary of the skills and their frequencies
    return dict(counter)
# End of the "softSkillsPerCat()" function

# --------------------------------------------------------------------- #

# Function that creates a co-occurence matrix per category and saves them in ".csv" files
def softCoOccurenceMatrix():
    # Create a list to hold the categories
    categories = []
    # Create a list to hold the skills
    skills = []
    # For every category in the dataframe's "category" column
    for c in set(data['category']):
        # Add the category to the "categories" list
        categories.append(c)
        # Get the dictionary of soft skills per category and their frequencies
        skills = softSkillsPerCat(c)
        # Create a dataframe containing the skills dictionary
        skills_frame = pd.DataFrame(skills.items(), columns=['Skill', c])
        # Sort skills by their frequency
        skills_frame.sort_values(by='Skill')
        # Save the dataframe as a ".csv" file
        skills_frame.to_csv('assets/co-occurences/{}.csv'.format(c))
# End of the "softCoOccurenceMatrix()" function

# ------------------------------------------------------------------- #

# Function that creates a plot of hard skills for a specific category
# This function also returns the skills and their respective frequencies
def overallHardSkills():
    # Create a list to hold all the skills
    allSkills = []
    # For every list of skills in the dataframe's "hard-skills" column
    for skill_list in data['hard-skills']:
        # Spill the contents of the list in the "allSkills" list
        allSkills += skill_list
    # Apply frequency distribution on the skills list 
    counter = FreqDist(allSkills)
    # Plot the list of skills depending on their frequencies 
    ## using the "plotSkills()" function
    print plotSkills(dict(counter), 'Overall Hard Skill Demand in Scotland')
# End of the "overallHardSkills()" function

# ------------------------------------------------------------------ #

# Function that creates a plot of hard skills of a specific category
# This function also returns the skills and their respective frequencies
def hardSkillsPerCat(category):
    # Extract the lists of hard skills for that specific category
    skills = data[data['category'] == category]['hard-skills']
    # Create a list to hold all the skill keywords
    allSkills = []
    # For every skill list in the extracted skills
    for skill_list in skills:
        # Spill the contents of the list in the "allSkills" list
        allSkills += skill_list
    # Apply frequency distribution on the skills list
    counter = FreqDist(allSkills)
    # Plot the list of skills depending on their frequencies
    ## using the "plotSkills()" function
    print plotSkills(dict(counter), 'Hard Skill Demand in Scotland for {}'.format(category))
    # Return a dictionary of the skills and their frequencies
    return dict(counter)
# End of the "hardSkillsPerCat()" function

# ------------------------------------------------------------------ #

# Function that creates a co-occurence matrix per category and saves them in ".csv" files
def hardCoOccurenceMatrix():
    # Create a list to hold the categories
    categories = []
    # Create a list to hold the skills
    skills = []
    # For every category in the dataframe's "category" column
    for c in set(data['category']):
        # Add the category to the "categories" list
        categories.append(c)
        # Get the dictionary of hard skills per category and their frequencies
        skills = hardSkillsPerCat(c)
        # Create a dataframe containing the skills dictionary
        skills_frame = pd.DataFrame(skills.items(), columns=['Skill', c])
        # Sort skills by their frequency
        skills_frame.sort_values(by='Skill')
        # Save the dataframe as ".csv" file
        skills_frame.to_csv('assets/co-occurences/{}.csv'.format(c))
# End of the "hardCoOccurenceMatrix()" function

# ------------------------------------------------------------------ #

if __name__ == '__main__':
    # Load data from file to dataframe
    data = pd.read_csv('assets/corpus/corpus.csv')
    # Print the data shape of the dataframe (rows x columns)
    print 'data shape: ', data.shape
    # Test datatype of columns
    # print data.dtypes

    # Print progress
    print('---')
    print('Transforming description to string...')
    data['descr'] = data['descr'].astype('str')
    
    # Call the drop duplicates function
    dropDuplicates()
    
    # Tokenize and add tokens to the dataframe
    tokenize()

    # APPLY FIRST APPROACH - MACHINE LEARNING
    
    # Calculate the average percentage of tokens extracted from the descriptions
    #getTokenAverage()
    
    # Create and display a popularity bar chart
    #popularityGraph(17, 10)

    # Print out the top tokens per job depending on their frequency 
    #topTokensPerJob(10)
    
    # Print all the categories and their respective top tokens
    #topTokensPerCat()

    # Create TF-IDF matrix
    #vectorizer, vz, tfidf = tfidfMatrix()

    # Create a word cloud with top lowest 40 tokens from the TF-IDF matrix
    #wordCloud(tfidf.sort_values(by=['tfidf'], ascending=True).head(40))

    # Create a word cloud with the top highest 40 tokens from the TF-IDF matrix
    #wordCloud(tfidf.sort_values(by=['tfidf'], ascending=False).head(40))

    # Create a TF-IDF histogram
    #tfidfHistogram(tfidf, 15, 7)

    # Print the top 30 tokens with the lowest TF-IDF scores
    #topLowestTfidf(tfidf, 30)

    # Print the top 30 tokens with the highest TF-IDF scores
    #topHighestTfidf(tfidf, 30)

    # Apply SVD to reduce the TF-IDF dimensions to 50
    #svd_tfidf = applySVD(vz)

    # Apply t-SNE to reduce the TF-IDF dimensions to two
    #tsne_tfidf = redDimension(svd_tfidf)

    # Plot the t-SNE dataframe on a scatter plot
    #groupTfidf(tsne_tfidf)

    # Apply and plot K-means with 34 clusters
    #applyKMeans(vz, 34)
    
    # Apply and plot LDA with 34 topics
    #applyLDA(34)

    # ---------------------------------------------------------------------------- #

    # APPLY SECOND APPROACH - PRE-DEFINED SETS OF SKILLS #

    # Save lists of soft skills in the dataframe where they match the tokens
    #getSoftSkills()

    # Save lists of hard skills in the dataframe where they match the tokens
    #getHardSkills()

    # Print top soft skills per every category
    #topSoftSkillsPerCat(10)

    # Print top hard skills per every category
    #topHardSkillsPerCat(10)

    # Draw a plot of the overall soft skills
    #overallSoftSkills()

    # Draw a plot of the overall hard skills
    #overallHardSkills()

    # Draw a plot of soft skills for the category "software developer"
    #softSkillsPerCat('software developer')

    # Draw a plot of hard skills for the category "software developer"
    #hardSkillsPerCat('software developer')

    # Create a co-occurence matrix of soft skills per category 
    ## for a network map in "Gephi"
    #softCoOccurenceMatrix()
    
    # Create a co-occurence matrix of hard skills per category
    ## for a network map in "Gephi"
    #hardCoOccurenceMatrix()
    
    # Show any drawings
    pylab.show()
