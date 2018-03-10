from bs4 import BeautifulSoup
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.probability import FreqDist

from string import punctuation
import requests, re, time, json

# Done with the help of goo.gl/yMaj4U
def getJobsAttr(url):
    try:
    	results_html = requests.get(url)
    except:
        return

    page_content = results_html.content
    results_soup = BeautifulSoup(page_content, 'lxml')

    jobs_list = []
    
    job_tags = results_soup.find_all('div', {'class': " row result"})
    
    for tag in job_tags:
    	job_title = tag.find('a', {'class': "turnstileLink"}).attrs['title']
        job_url   = tag.find('a').get('href')
        job_date  = tag.find('span', {'class': "date"}).getText()

	job_attrs = {}

        job_attrs['job_date']  = job_date
	job_attrs['job_url']   = job_url
        job_attrs['job_title'] = job_title   

	jobs_list.append(job_attrs) 
    
    return jobs_list


# Done with the help of the 'text_cleaner()' method 
# found at goo.gl/2DZMtG
def cleanJobText(url):
    try:
        job_page = requests.get(url)
    except:
        return

    # Get the html of the job page
    job_html = BeautifulSoup(job_page.text, "lxml")

    # Remove script elements from the html
    for script in job_html(["script", "style"]):
        script.extract()

    # Get the text from the html 
    job_text = job_html.get_text()

    # Break text into lines
    lines = (line.strip() for line in job_text.splitlines())

    # Break lines into chunks 
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # Fix spacing issues
    def chunkSpace(chunk):
        final_chunk = chunk + '  '
        return final_chunk

    job_text = ''.join(chunkSpace(chunk) for chunk in chunks if chunk).encode('utf-8')

    # Clean unicode
    try:
        job_text = job_text.decode('unicode_escape', 'ascii').encode('ascii', 'ignore')
    except:
        return

    # Drop symbols
    job_text = re.sub("[^a-zA-Z.+]", "  ", job_text)

    # Lowercase and split into words
    job_tokens = job_text.lower().split()

    # Filter out stop words
    stop_words = stopwords.words("english")

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
    stop_words.append(".")
    stop_words.append("+")

    stop_words = set(stop_words)

    job_tokens = [word for word in job_tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    job_tokens = [lemmatizer.lemmatize(word) for word in job_tokens]

    # Stem
    #stemmer = PorterStemmer()

    #job_tokens = [stemmer.stem(word) for word in job_tokens]
    
    encoded = [word.encode('utf-8', 'strict') for word in job_tokens]

    return encoded

# Get word frequency distribution
def freqDist(tokens):
    return FreqDist(tokens).most_common()


def findJobs(starting_page = 0, pages_limit = 1, location = 'Aberdeen', query = 'Software Developer'):

    # Format query to fit url
    # Replace spaces with '+' and lowercase
    query_format    = re.sub('  ', '+', query).lower()
    location_format = re.sub('  ', '+', location)
    base_url        = 'https://www.indeed.co.uk/jobs?q={0}&l={1}&start='.format(query_format, location_format)

    jobs_dict    = defaultdict(list)
    jobs_counter = 0

    for page in xrange(starting_page, starting_page + pages_limit):
        print '\n', 'Extracting page ' + str(page+1), '\n'

        # Extract job attributes of jobs on current page
        jobs_attr = getJobsAttr(base_url + str(page*10))

        # For every job on the page
        for job in xrange(0, len(jobs_attr)):
            job_title = jobs_attr[job]['job_title'].lower()
            job_url   = jobs_attr[job]['job_url']
            job_date  = jobs_attr[job]['job_date']

            jobs_counter += 1

            print "Job " + str(jobs_counter) + ": " + repr(job_title) + ', ' + repr(job_url)

            job_tokens = cleanJobText('http://indeed.co.uk' + job_url)
            top_ten_tk = freqDist(job_tokens) 

            job_dict  = {}

            job_dict['job_tokens'] = top_ten_tk
            job_dict['job_date']   = job_date
            job_dict['job_url']    = job_url

            jobs_dict[job_title].append(job_dict)

            time.sleep(1)

    print 'Extraction successful with a total of {0} jobs'.format(jobs_counter)

    #Save to json
    filename = query.lower() + ".json"
    file = open(filename, "w")
    json.dump(jobs_dict, file, indent=4)
    file.close

# Test getJobsAttr()
print getJobsAttr('https://www.indeed.co.uk/jobs?q=software+developer&l=Aberdeen&start=0')[0]

# Test cleanJobText()
print cleanJobText('https://www.indeed.co.uk/cmp/Insiso-Ltd/jobs/Net-Software-Developer-57266562ff84a36a?q=software+developer')

# Test findJobs()
findJobs()
