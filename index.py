from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import stopwords
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
    def chunk_space(chunk):
        final_chunk = chunk + '  '
        return final_chunk

    job_text = ''.join(chunk_space(chunk) for chunk in chunks if chunk).encode('utf-8')

    # Clean unicode
    try:
        job_text = job_text.decode('unicode_escape').encode('ascii', 'ignore')
    except:
        return

    # Drop symbols
    job_text = re.sub("[^a-zA-Z.+3]", "  ", job_text)
    
    # Lowercase and split into words
    job_text = job_text.lower().split()

    # Filter out stop words
    stop_words = set(stopwords.words("english"))
    job_text   = [word for word in job_text if word not in stop_words]

    # Create a set of words 
    # job_text = list(set(job_text))

    return job_text


def findJobs(starting_page = 0, pages_limit = 20, location = 'Aberdeen', query = 'Software Developer'):

    # Format query to fit url
    # Replace spaces with '+' and lowercase
    query_format    = re.sub('  ', '+', query).lower()
    location_format = re.sub('  ', '+', location)
    base_url        = 'https://www.indeed.co.uk/jobs?q={0}&l={1}&start='.format(query_format, location_format)

    jobs_dict    = defaultdict(list)
    jobs_counter = 0

    for page in xrange(starting_page, starting_page + pages_limit):
        print 'URL: ' + base_url + str(page*10), '\n'

        # Extract job attributes of jobs on current page
        jobs_attr = getJobsAttr(base_url + str(page*10))

        # For every job on the page
        for job in xrange(0, len(jobs_attr)):
            job_title = jobs_attr[job]['job_title'].lower()
            job_url   = jobs_attr[job]['job_url']
            job_date  = jobs_attr[job]['job_date']

            print repr(job_title) + ', ' + repr(job_url)

            job_text  = cleanJobText('http://indeed.co.uk' + job_url)
            
            job_dict  = {}

            job_dict['job_text'] = job_text
            job_dict['job_date'] = job_date
            job_dict['job_url']  = job_url

            jobs_dict[job_title].append(job_dict)

            jobs_counter = jobs_counter + 1

            time.sleep(1)

    print 'Extraction successful with a total of {0} jobs'.format(jobs_counter)

    #Save to json
    filename = query.lower() + ".json"
    file = open(filename, "a")
    json.dump(jobs_dict, file, indent=4)
    file.close

# Test getJobsAttr()
print getJobsAttr('https://www.indeed.co.uk/jobs?q=software+developer&l=Aberdeen&start=0')[0]

# Test cleanJobText()
print cleanJobText('https://www.indeed.co.uk/cmp/Insiso-Ltd/jobs/Net-Software-Developer-57266562ff84a36a?q=software+developer')[:20]

#Test findJobs()
findJobs()
