import requests
from bs4 import BeautifulSoup
import pandas as pd
url = "https://pokemondb.net/pokedex/stats/gen1"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'class': 'data-table'})
headers = []
for th in table.find('thead').find_all('th'):
    headers.append(th.text.strip())

rows = []
for tr in table.find('tbody').find_all('tr'):
    cells = []
    for td in tr.find_all('td'):
        cells.append(td.text.strip())
    rows.append(cells)

df = pd.DataFrame(rows, columns = headers)
df.to_csv('gen1_pokemon_stats.csv', index = False)
print("File gen1_pokemon_stats.csv has been initialized.")
