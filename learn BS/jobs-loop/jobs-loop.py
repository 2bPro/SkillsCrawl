from bs4 import BeautifulSoup
import datetime
import requests
import time	
import json
import re

def getJobData(url):
    try:
        job_html = requests.get(url)
    except:
        return 0

    job_content = job_html.content
    job_soup = BeautifulSoup(job_content, "lxml")
    
    tags = job_soup.find_all("div", {"data-tn-component": "organicJob"})
    companies_list = [x.span.text for x in tags]
    job_attributes = [x.h2.a.attrs for x in tags]
    dates = [x.find_all("span", {"class": "date"}) for x in tags]

    [job_attributes[i].update({"company": companies_list[i].strip()}) for i, x in enumerate(job_attributes)]
    [job_attributes[i].update({"date posted": dates[i][0].text.strip()}) for i, x in enumerate(job_attributes)]

    return job_attributes


def getJobDetails(url):
    try:
        job_html = requests.get(url)
    except:
        return 0

    job_content = job_html.content
    job_soup = BeautifulSoup(job_content, "html.parser")

    #Get job title
    b_tag = job_soup.find("b", {"class":"jobtitle"})
    title = b_tag.font.text  

    #Get job location
    location = job_soup.find("span", {"class":"location"})
    
    #Get job description
    descriptions_list = []

    td_tag = job_soup.find("td", {"class":"snip"})
    
    for ul in td_tag.find_all("ul"):
        for li in ul.findAll("li"):
            descriptions_list.append(li.text)

    #Concatenate all descriptions in list to a string
    description = ""

    for d in descriptions_list:
        description += d + "  "

    #Get job date
    time = job_soup.find("span", {"class":"date"})    

    #Append collected data to dictionary
    attr_list = {}

    attr_list["location"] = location.get_text()
    attr_list["description"] = description
    attr_list["title"] = title

    return attr_list


def findJobs(starting_page = 0, pages_limit = 10, location = 'Aberdeen', query = 'IT'):
	query_formatted = re.sub(' ', '+', query)
	location_formatted = re.sub(' ', '+', location)
	base_url = 'https://www.indeed.co.uk/jobs?q={0}&l={1}&start='.format(query_formatted, location_formatted)
	global jobs_counter
	jobs_counter = 0
	jobs_list = []

	for page in xrange(starting_page, starting_page + pages_limit):
		print 'URL: {0}'.format(base_url + str(page*10)), '\n'

		# extract job data from indeed search results page
		job_attributes = getJobData(base_url + str(page*10))

		for job in xrange(0, len(job_attributes)):
			title = job_attributes[job]['title']
			href  = job_attributes[job]['href']

			print '{0}, {1}'.format(repr(title), repr(href))

			job_details = getJobDetails('http://indeed.co.uk' + href)

			jobs_list.append(job_details)

			jobs_counter = jobs_counter + 1
	
			time.sleep(5)

	print 'Extraction successful with a total of {0} jobs'.format(jobs_counter)

	#Save to json
	file = open("jobs.json", "w")
    	json.dump(jobs_list, file)
	file.close


findJobs()
			


    
