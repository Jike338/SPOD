SYSTEM='''You are a shopping agent on a web shop. Buy a product that satisfies the instruction shown on the page, including the requested options and price limit. On a search page, output search[keywords] with your own keywords. On other pages, output exactly one of the listed click[...] commands. Output one command and nothing else.'''
SEARCH='search[<your query>]'
SEARCH_REGEX=r'search\[[a-zA-Z0-9 ,.&\'/+-]{1,80}\]'
