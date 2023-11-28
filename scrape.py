import requests
from bs4 import BeautifulSoup
import csv
import re

url_list = [
    'https://iusf.indiana.edu/little500/results.html?raceType=Little+500+Race&year=All&gender=All&teamName=&l500submit=Search#results',
    'https://iusf.indiana.edu/little500/results.html?raceType=Qualifications&year=All&gender=All&teamName=&l500submit=Search#results',
    'https://iusf.indiana.edu/little500/results.html?raceType=Team+Pursuit&year=All&gender=All&teamName=&l500submit=Search#results',
    'https://iusf.indiana.edu/little500/results.html?raceType=Individual+Time+Trials&year=All&gender=All&teamName=&RiderName=&l500submit=Search#results',
    'https://iusf.indiana.edu/little500/results.html?raceType=Rookie+of+the+Year&year=All&gender=All&teamName=&RiderName=&l500submit=Search#results',
    'https://iusf.indiana.edu/little500/results.html?raceType=All+Star+Rider&year=All&gender=All&teamName=&RiderName=&l500submit=Search#results'
]

for url in url_list:
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table')

        title_match = re.search(r'raceType=(.*?)&', url)
        title = title_match.group(1) if title_match else "unknown"

        title = re.sub(r'[^a-zA-Z0-9]', '_', title)

        with open(f'{title}.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            for row in table.find_all('tr'):
                row_data = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                writer.writerow(row_data)

        print(f'Table data from {url} has been scraped and saved to {title}.csv')
    else:
        print(f'Failed to retrieve the web page: {url}')
