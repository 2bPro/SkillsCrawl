from bs4 import BeautifulSoup
import requests
import json

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

    #Append collected data to dictionary
    attr_list = {}

    attr_list["location"] = location.get_text()
    attr_list["description"] = description
    attr_list["title"] = title

    #Save to json
    file = open("job.json", "w")
    json.dump(attr_list, file)
    file.close
