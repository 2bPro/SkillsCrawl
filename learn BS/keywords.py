from bs4 import BeautifulSoup

import requests

def analyseJob(url):
    try:
        job_html = requests.get(url)
    except:
        return 0

    job_content = job_html.content

    job_soup = BeautifulSoup(job_content, "lxml")

    job_body = job_soup('body')[0]

    job_text = job_body.text

    c_count       = job_text.count('C#')      + job_text.count('c#')
    sql_count     = job_text.count('SQL')     + job_text.count('sql')
    angular_count = job_text.count('Angular') + job_text.count('angular')

    print 'C# count: {0}, SQL count: {1}, Angular count: {2}'.format(c_count, sql_count, angular_count)


analyseJob('https://www.indeed.co.uk/cmp/Motion-Software/jobs/Software-Developer-f9e08782cc770ce8?q=Software+Developer')
