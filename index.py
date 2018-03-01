from bs4 import BeautifulSoup
import requests

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

#Extract data of first job of search results
print getJobsAttr('https://www.indeed.co.uk/jobs?q=software+developer&l=Aberdeen&start=0')
