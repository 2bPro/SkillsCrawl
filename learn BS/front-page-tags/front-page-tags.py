from bs4 import BeautifulSoup
import requests
import json

def getJobData(url):
    try:
        job_html = requests.get(url)
    except:
        return 0

    job_content = job_html.content
    job_soup = BeautifulSoup(job_content, "lxml")
    
    tags = job_soup.find_all("div", {"data-tn-component": "organicJob"})
    companies_list = [x.span.text for x in tags]
    attrs_list = [x.h2.a.attrs for x in tags]
    dates = [x.find_all("span", {"class": "date"}) for x in tags]

    [attrs_list[i].update({"company": companies_list[i].strip()}) for i, x in enumerate(attrs_list)]
    [attrs_list[i].update({"date posted": dates[i][0].text.strip()}) for i, x in enumerate(attrs_list)]

    file = open("jobs.json", "w")
    json.dump(attrs_list, file)
    file.close

    return attrs_list

#Extract data of first job of search results
print getJobData('https://www.indeed.co.uk/Software-Developer-jobs-in-Aberdeen')[0]
    
