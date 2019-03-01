system('wget -O nfl.json "http://api.fantasy.nfl.com/v1/players/stats?statType=seasonStats&season=2010&week=1&format=json"')
library(jsonlite)
mydf <- fromJSON('nfl.json')
class(mydf)
printhead(mydf,13))

