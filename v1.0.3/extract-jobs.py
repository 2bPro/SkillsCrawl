from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import requests, re, pandas as pd

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
def createDataFrame(starting_page = 0, pages_limit = 1, location = 'Aberdeen'):
    job_categories  = ['software developer', 'networking', 'tech support']
    location_format = re.sub('  ', '+', location)

    jobs_list = []

    for category in job_categories:
        print "Searching for category: " + category, '\n'

        category_format = re.sub('  ', '+', category).lower()
        base_url        = 'https://www.indeed.co.uk/jobs?q={0}&l={1}&start='.format(category_format, location_format)
        
        jobs_counter    = 0

        for page in xrange(starting_page, starting_page + pages_limit):
            print '\n', 'Extracting page ' + str(page+1), '\n'
            jobs_attr =  getJobsAttr(base_url + str(page*10))

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
                jobs_counter += 1
	
    headers = ['category', 'title', 'date', 'url', 'descr']
    jobs    = pd.DataFrame(jobs_list, columns=headers)

    with open('jobs.csv', 'a') as f:
        jobs.to_csv(f, header=True, encoding='utf-8', index=False)

    print(jobs)


if __name__ == '__main__':
    createDataFrame()
