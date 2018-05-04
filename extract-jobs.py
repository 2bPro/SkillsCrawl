# Import necessary libraries
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import requests, re, pandas as pd, math, time

# Function that returns the number of resulted jobs
# It takes as input parameter the URL of the first resulted page
# Created with the help of Beautifulsoup and Python docs
def getJobsNo(url):
    try:
        results_html = requests.get(url)
    except:
        return

    # Extract the page's content
    page_content = results_html.content
    # Create a Python object tree
    results_soup = BeautifulSoup(page_content, 'lxml')
    # Search for the "searchCount" HTML tag and extract contents
    no_jobs = results_soup.find_all('div', {'id': "searchCount"})
    # Convert the contents to "String" type
    no_jobs = str(no_jobs)
    # Clean unicode symbols
    no_jobs = no_jobs.decode('unicode_escape', 'ascii').encode('ascii', 'ignore')
    # Clean any remaining symbols and words
    no_jobs = re.sub(r'[a-zA-Z/<>"=]', '  ', no_jobs)
    # Clean any commas
    no_jobs = no_jobs.replace(',', '')
    # Convert the remaining digits to "Integer" type
    no_jobs = [int(s) for s in no_jobs.split() if s.isdigit()]
    
    # Return the number that represents the total number of jobs
    try:
        return no_jobs[2]
    except:
        return
# End of the "getJobsNo()" function

# --------------------------------------------------------------------------------- #

# Function that returns the jobs from a results page and their attributes
# It takes as input parameter the URL of a resulted page
# Created with the help of the article found at goo.gl/yMaj4U and "Beautifulsoup" docs
def getJobsAttr(url):
    try:
    	results_html = requests.get(url)
    except:
        return
    # Extract the page's content
    page_content = results_html.content
    # Create a Pyhon object tree
    results_soup = BeautifulSoup(page_content, 'lxml')
    # Create a list to hold the jobs of a page and their attributes
    jobs_list = []
    # The jobs are returned and displayed as rows
    # Find the " row result" HTML tag and extract the rows
    job_tags = results_soup.find_all('div', {'class': " row result"})
    
    # For every tag in the extracted rows
    for tag in job_tags:
        # Find the HTML tag corresponding to the job's title
    	job_title = tag.find('a', {'class': "turnstileLink"}).attrs['title']
        # Find the HTML tag corresponding to the job's URL
        job_url   = tag.find('a').get('href')
        # Find the HTML tag corresponding to the job's post date
        job_date  = tag.find('span', {'class': "date"}).getText()
        # Find the HTML tag corresponding to the job's location
        job_loc   = tag.find('span', {'class': "location"}).getText()
        
        # Create a job object
	job_attrs = {}
        # Add the attributes to the object
        job_attrs['job_date']  = job_date
	job_attrs['job_url']   = job_url
        job_attrs['job_loc']   = job_loc
        job_attrs['job_title'] = job_title   
        # Add the job object tp a list of jobs
	jobs_list.append(job_attrs) 
    # Return the list of jobs
    return jobs_list
# End of the "getJobAttr()" function

# ---------------------------------------------------------------------------------- #

# Function that returns the pure text of a specific job advertisement page
# It takes as input parameter the URL attribute of a job object
# Created with the help of the article found at goo.gl/2DZMtG
def getJobText(url):
    try:
        job_page = requests.get(url)
    except:
        return

    # Extract the page's content
    page_content = job_page.text
    # Create a Python object tree
    job_html = BeautifulSoup(job_page.text, "lxml")
    
    # Remove script elements from the html
    for script in job_html(["script", "style"]):
        script.extract()
    # Extract the text from the html 
    job_text = job_html.get_text()
    # Fragment the text into lines
    lines = (line.strip() for line in job_text.splitlines())
    # Fragment the lines into chunks
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Fix spacing issues between chunks
    def chunkSpace(chunk):
        final_chunk = chunk + '  '
        return final_chunk
    # Encode the text
    job_text = ''.join(chunkSpace(chunk) for chunk in chunks if chunk).encode('utf-8')
    # Clean unicode
    try:
        job_text = job_text.decode('unicode_escape', 'ascii').encode('ascii', 'ignore')
    except:
        return
    # Return the job text
    return job_text
# End of the "getJobText()" function

# ---------------------------------------------------------------------------------- #

# Function that returns the actual date of the posting of an advertisement
# It takes as input the date attribute of a job object
# Created with the help of Python datetime docs
def getDate(string):
    # Remove symbols
    string = re.sub(r'[^\w]', '  ', string)
    # Extract numbers
    number = int(string.split()[0])
    # Extract text
    string = ''.join(i for i in string if not i.isdigit())
    # Remove text whitespace
    string = string.replace("  ", "")
    # Extract the first character
    char   = string[:1]
    # Get the current date
    date   = datetime.now()
    
    # If the character is "d"
    if(char === "d"):
        # The number is extracted as days from today's date
        date = date - timedelta(days=number)

    # Indeed only returns advertisements posted in the last 30 days
    # This is why months will not be taken in consideration

    # Convert the date into "String" type
    date  = date.strftime("%d/%m/%y")
    # Return the date
    return date
# End of the "getDate()" function

# --------------------------------------------------------------------------------- #

# Function that extracts the job details, creates a dataframe and exports it as ".csv" file 
# It takes as input parameters two constant variables used for the manufacturing of a URL
# Created with the help of Python and Pandas docs
def createDataFrame(starting_page = 0, location = 'Scotland'):
    # List of job categories to be extracted 
    job_categories  = ['application developer', 
                       'business intelligence analyst',
                       'computer scientist',
                       'data analyst', 'database administrator', 'data scientist',
                       'graphic web designer',
                       'help desk technical support',
                       'j2ee developer',
                       'net developer',
                       'network engineer',
                       'robotics software engineer',
                       'software developer', 'software engineer', 'systems analyst',
                       'unix engineer',
                       'web developer'
                      ]
    
    # Format the location constant for usage in the URL
    location = re.sub('  ', '+', location)
    # Create a list of extracted jobs
    jobs_list = []
    # Set a counter for the total number of extractions
    total_count = 0
    
    # For every job category in the list of categories
    for category in job_categories:
        # Print the category to be extracted to announce extraction progress
        print "Searching for category: " + category, '\n'
        # Format the category for usage in the URL
        category = re.sub('  ', '+', category).lower()
        # Manufacture a URL using the category as query
        base_url = 'https://www.indeed.co.uk/jobs?q={0}&l={1}&start='.format(category, location)
        # Obtain the number of jobs to be extracted
        pages_limit = getJobsNo(base_url)
        # Set a counter for the total number of extractions per category
        cat_count = 0
        # For every page in range between the starting page and the available number of pages
        for page in xrange(starting_page, starting_page + pages_limit):
            # Obtain the jobs of the current page and their attributes
            jobs_attr =  getJobsAttr(base_url + str(page*10))
            # If there are attributes and the extraction of a job is possible
            if jobs_attr:
                # Print the page that is to be extracted
                print '\n', 'Extracting page ' + str(page+1), '\n'
                # For every job on the current page
                for job in xrange(0, len(jobs_attr)):
                    # Set the usual job attributes
                    job_title = jobs_attr[job]['job_title'].lower()
                    job_url   = jobs_attr[job]['job_url']
                    job_date  = getDate(jobs_attr[job]['job_date'])
                    job_loc   = jobs_attr[job]['job_loc']
                    # As well as the extracted text 
                    job_descr = getJobText('http://indeed.co.uk' + job_url)
                    # Create a list of attributes for the current job
                    job_attrs = []
                    # And append all the attributes, including category and description
                    job_attrs.append(category)
		    job_attrs.append(job_title)
		    job_attrs.append(job_date)
                    job_attrs.append(job_loc)
		    job_attrs.append(job_url)
		    job_attrs.append(job_descr)
                    # Add the job attributes list to a list of jobs
                    jobs_list.append(job_attrs)
                    # Increment the counter of extracted jobs per category
                    cat_count += 1
                    # Pause the extraction one second
                    time.sleep(1)
                # Continue extraction of jobs until the whole results page is covered
        # Change the URL and start extracting the next page

        # Print the total number of extractions per category
        print("Successfully extracted {0} jobs for the {1} job title").format(cat_count, category)
        print("-----------------------------------------------------")
        # Add that number to the total number of extractions overall
        total_count += cat_count
	
        # Prepare the dataframe column names
        headers = ['category', 'title', 'date', 'location', 'url', 'descr']
        # Create a dataframe from the list of jobs and their attributes
        jobs    = pd.DataFrame(jobs_list, columns=headers)
        # Create a ".csv" file with the name of the category
        # Save the dataframe into the file
        with open('assets/corpus/{0}.csv'.format(category), 'w') as f:
            jobs.to_csv(f, header=True, encoding='utf-8', index=False)
        
        # After all the categories have been extracted, print the total number of extractions
        print("Extraction successful with a total number of {0} jobs").format(total_count)

if __name__ == '__main__':
    # Call the extraction function
    createDataFrame()
