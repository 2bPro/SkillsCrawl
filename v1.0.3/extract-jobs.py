from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import requests, re, pandas as pd, math

def getJobsNo(url):
    try: 
        results_html = requests.get(url)
    except:
        return

    page_content = results_html.content
    results_soup = BeautifulSoup(page_content, 'lxml')

    no_jobs = results_soup.find_all('div', {'id': "searchCount"})
    no_jobs = str(no_jobs)
    no_jobs = no_jobs.decode('unicode_escape', 'ascii').encode('ascii', 'ignore')
    no_jobs = re.sub(r'[a-zA-Z/<>"=]', '  ', no_jobs)
    no_jobs = no_jobs.replace(',', '')
    no_jobs = [int(s) for s in no_jobs.split() if s.isdigit()]

    if int(no_jobs[2]) > 1000:
        return 100
    else:
        return no_jobs[2]/10

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
def getJobText(url):
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

    return job_text

# Python datetime documentation
def getDate(string):
    # Remove symbols
    string = re.sub(r'[^\w]', '  ', string)

    # Extract numbers
    number = int(string.split()[0])

    # Remove numbers
    string = ''.join(i for i in string if not i.isdigit())

    # Remove whitespace
    string = string.replace("  ", "")

    # Get first character
    char   = string[:1]

    # Get current date
    date   = datetime.now()

    if char == "d":
        date = date - timedelta(days=number)

    date  = date.strftime("%d/%m/%y")

    return date

# Pandas documentation
def createDataFrame(starting_page = 0, location = 'Scotland'):
    job_categories  = ['analyst', 'application developer', 
                       'business analyst', 
                       'computer scientist',
                       'data analyst', 'database administrator',
                       'graphic web designer', 
                       'help desk technical support', 'help desk technician',
                       'informatica developer', 'intelligence analyst', 'it auditor', 'it specialist',
                       'j2ee developer', 'java developer',
                       'kronos programmer', 
                       'net developer', 'network engineer', 
                       'oracle developer', 
                       'programmer analyst', 
                       'research analyst',
                       'software developer', 'software engineer', 'systems analyst', 'software test engineer', 
                       'unix engineer', 
                       'web developer', 'web programmer', 
                       'xml developer', 'xsd developer']

    location_format = re.sub('  ', '+', location)

    jobs_list = []
    total_count = 0

    for category in job_categories:
        print "Searching for category: " + category, '\n'

        category_format = re.sub('  ', '+', category).lower()
        base_url        = 'https://www.indeed.co.uk/jobs?q={0}&l={1}&start='.format(category_format, location_format)
        pages_limit     = getJobsNo(base_url + "0")
        
        cat_count       = 0

        for page in xrange(starting_page, starting_page + pages_limit):
            jobs_attr =  getJobsAttr(base_url + str(page*10))

            if jobs_attr:
                print '\n', 'Extracting page ' + str(page+1), '\n'
                for job in xrange(0, len(jobs_attr)):
                    job_title = jobs_attr[job]['job_title'].lower()
                    job_url   = jobs_attr[job]['job_url']
                    job_date  = getDate(jobs_attr[job]['job_date'])
                    job_descr = getJobText('http://indeed.co.uk' + job_url)
                
                    job_attrs = []
                    job_attrs.append(category)
		    job_attrs.append(job_title)
		    job_attrs.append(job_date)
		    job_attrs.append(job_url)
		    job_attrs.append(job_descr)

                    jobs_list.append(job_attrs)
                    cat_count += 1

        print("Successfully extracted {0} jobs for the {1} job title").format(cat_count, category)
        print("-----------------------------------------------------")
        total_count += cat_count
	
    headers = ['category', 'title', 'date', 'url', 'descr']
    jobs    = pd.DataFrame(jobs_list, columns=headers)

    with open('jobs.csv', 'w') as f:
        jobs.to_csv(f, header=True, encoding='utf-8', index=False)

    print("Extraction successful with a total number of {0} jobs").format(total_count)

if __name__ == '__main__':
    createDataFrame()
