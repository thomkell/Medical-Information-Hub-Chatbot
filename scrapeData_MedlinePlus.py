# /Users/thomaskeller/Dropbox/Projects
# 16. Aug 2024
# T.Keller
# /Users/thomaskeller/Library/Mobile\ Documents/com~apple~CloudDocs/Projects/HealthcareChatbot

import requests
from bs4 import BeautifulSoup
import pandas as pd

# List of URLs for MedlinePlus Health Topics A-Z pages
base_urls = [
    "https://medlineplus.gov/healthtopics_a.html",
    "https://medlineplus.gov/healthtopics_b.html",
    "https://medlineplus.gov/healthtopics_c.html",
    "https://medlineplus.gov/healthtopics_d.html",
    "https://medlineplus.gov/healthtopics_e.html",
    "https://medlineplus.gov/healthtopics_f.html",
    "https://medlineplus.gov/healthtopics_g.html",
    "https://medlineplus.gov/healthtopics_h.html",
    "https://medlineplus.gov/healthtopics_i.html",
    "https://medlineplus.gov/healthtopics_j.html",
    "https://medlineplus.gov/healthtopics_k.html",
    "https://medlineplus.gov/healthtopics_l.html",
    "https://medlineplus.gov/healthtopics_m.html",
    "https://medlineplus.gov/healthtopics_n.html",
    "https://medlineplus.gov/healthtopics_o.html",
    "https://medlineplus.gov/healthtopics_p.html",
    "https://medlineplus.gov/healthtopics_q.html",
    "https://medlineplus.gov/healthtopics_r.html",
    "https://medlineplus.gov/healthtopics_s.html",
    "https://medlineplus.gov/healthtopics_t.html",
    "https://medlineplus.gov/healthtopics_u.html",
    "https://medlineplus.gov/healthtopics_v.html",
    "https://medlineplus.gov/healthtopics_w.html",
    "https://medlineplus.gov/healthtopics_xyz.html"
]

all_topic_links = []

# Step 1: Extract links from alphabetical pages
for url in base_urls:
    print(f"Scraping topic links from {url}...")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find all <li> elements with class "item"
    topic_items = soup.find_all("li", {"class": "item"})
    
    for item in topic_items:
        link = item.find("a", href=True)
        if link:
            full_url = link['href']
            topic_name = link.text.strip()
            all_topic_links.append({"Topic": topic_name, "URL": full_url})

print(f"Total topics found: {len(all_topic_links)}")

# Convert the collected links to a DataFrame
df_links = pd.DataFrame(all_topic_links)

# Step 2: Scrape summaries for each topic
all_details = []

for index, row in df_links.iterrows():
    print(f"Scraping details for {row['Topic']}...")
    try:
        response = requests.get(row['URL'])
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Initialize a dictionary to store the details
        details = {
            "URL": row['URL'],
            "Topic": row['Topic'],
            "Summary": ""
        }
        
        # Extract the summary from the correct section
        summary_section = soup.find("div", {"id": "topic-summary", "class": "syndicate"})
        if summary_section:
            summary_paragraphs = summary_section.find_all("p")
            summary_text = "\n".join([p.text.strip() for p in summary_paragraphs])
            details["Summary"] = summary_text
        else:
            print(f"No summary section found for {row['URL']}.")
        
        all_details.append(details)
        
    except Exception as e:
        print(f"Failed to scrape {row['Topic']} at {row['URL']}: {e}")

# Convert the list of dictionaries to a DataFrame
details_df = pd.DataFrame(all_details)

# Save the detailed information to a CSV file
details_df.to_csv("medlineplus_health_topic_details_summary.csv", index=False)

print("Scraping complete. Details saved to medlineplus_health_topic_details_summary.csv.")
