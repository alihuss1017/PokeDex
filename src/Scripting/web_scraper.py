import requests
from bs4 import BeautifulSoup
import pandas as pd


url = "https://pokemondb.net/pokedex/stats/gen1"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# FIND STATS CATEGORIES
table = soup.find('table', {'class': 'data-table'})
headers = []
for th in table.find('thead').find_all('th'):
    headers.append(th.text.strip())

#FIND STATS VALUES
rows = []
for tr in table.find('tbody').find_all('tr'):
    cells = []
    for td in tr.find_all('td'):
        cells.append(td.text.strip())
    rows.append(cells)

df = pd.DataFrame(rows, columns = headers)

# FIND POKEDEX ENTRIES
poke_data = []
for number in range(1, 152):
    URL = 'https://pokemondb.net/pokedex/' + str(number)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    entry = soup.find_all('td', class_ = 'cell-med-text')
    poke_data.append(entry[0].get_text(strip = True))

poke_df = pd.DataFrame(poke_data)
df = pd.concat([df, poke_df], axis = 1, ignore_index = False)
df.to_csv('pokedex_info.csv', index = False)
print("pokedex_info.csv file initialized.")