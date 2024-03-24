#code from terminal
#scrapy startproject project
#cd project
#scrapy genspider Kowalski_Wlodarczyk_Scrapy

import scrapy
import re

class FirstSpider(scrapy.Spider):
    name = "Kowalski_Wlodarczyk_Scrapy"

    def start_requests(self):
        # Generate browse pages
        for i in range(1, 26):  
            url = "https://app.thestorygraph.com/browse?page=" + str(i)
            yield scrapy.Request(url=url, callback=self.parse_browse_results)

    def parse_browse_results(self, response):
        # Extract book links from browse results
        for a in response.css('a'):
            href = a.attrib.get('href')
            if '/books/' in href and 'edition' not in href:
                yield response.follow(href, self.parse_book)

    def parse_book(self, response):
        # Get title and author 
        title_author=response.xpath("/html/head/title/text()").extract()
        title_author=re.findall(r'.+(?= \| )',title_author[0])[0]
        # Get number of pages
        try:
            pages=response.xpath("/html/body/div[1]/div/main/div/div[3]/div/div[2]/p/text()[1]").extract()
            pages=int(re.findall(r'[0-9]+(?= pages)',pages[0])[0])
        except:
            pages='audiobook'
        # Get description
        genre=[]
        genre=response.xpath("/html/body/div[1]/div/main/div/div[3]/div/div[2]/div[4]/div/span/text()").extract()
        genres=['art','autobiography','biography','business','childrens','classics','comics','computer science','contemporary','cookbook','crime',
                'design','dystopian','economics','education','erotica','essays','fantasy','feminism','fiction','food and drink','gender','graphic novels','health','historical',
                'lgbtqia+','literary','magical realism','manga','mathematics','memoir','middle grade','music','mystery','nature','nonfiction','philosophy',
                'play','poetry','politics','psychology','race','reference','religion','romance','science','science fiction','self help',
                'short stories','sociology','speculative fiction','sports','technology','thriller','travel','true crime','video games','young adult']
        # Remove not-genres
        for i in genre:
            if i not in genres:
                genre.remove(i)
        moods=['adventurous','challenging','dark','emotional','funny','hopeful', 'informative','inspiring',
                'lighthearted','mysterious','reflective','relaxing','sad','tense']
        # Remove not-genres
        for i in genre:
            if i in moods:
                genre.remove(i)
        paces=['medium-paced','slow-paced','fast-paced']
        # Remove not-genres
        for i in genre:
            if i in paces:
                genre.remove(i)
        # Get moods       
        mood = response.xpath('/html/body/div[1]/div/main/div/div[3]/div/div[2]/div[8]/div[1]/div[1]/div/span/text()').extract()
        # Converting list to dictionary
        def convert(lst):
            res_dict = {}
            for i in range(0, len(lst), 2):
                res_dict[lst[i]] = lst[i + 1]
            return res_dict
        mood = convert(mood)
        # Get average rating
        try:
            stars = float(response.xpath('/html/body/div[1]/div/main/div/div[3]/div/div[2]/div[8]/span/text()').extract()[0])
        except:
            stars = float(response.xpath('/html/body/div[1]/div/main/div/div[3]/div/div[2]/div[9]/span/text()').extract()[0])
        # Get number of reviews
        try:
            reviews=response.xpath('/html/body/div[1]/div/main/div/div[3]/div/div[2]/div[8]/h3/span/a/text()').extract()
            reviews=int(re.findall(r'[0-9]+,?[0-9]+(?= reviews)',reviews[0])[0].replace(",", ""))
        except:
            reviews=response.xpath('/html/body/div[1]/div/main/div/div[3]/div/div[2]/div[9]/h3/span/a/text()').extract()
            reviews=int(re.findall(r'[0-9]+,?[0-9]+(?= reviews)',reviews[0])[0].replace(",", ""))
        # Check if book is of a series
        series=0
        for a in response.css('a'):
            href = a.attrib.get('href')
            if ('/series/' in href):
                series=1
        # Get additional information
        try:
            additional_info=response.xpath('/html/body/div[1]/div/main/div/div[3]/div/div[2]/div[8]/div[2]/span[1]/text()').extract()[0]
            mix=re.findall(r"(?<=A mix: )([0-9]+\%)",additional_info)[0]
            character=re.findall(r"(?<=Character: )([0-9]+\%)",additional_info)[0]
            plot=re.findall(r"(?<=Plot: )([0-9]+\%)",additional_info)[0]
        except:
            mix=0
            plot=0
            character=0
        # Saving results
        yield {'title_author':title_author, 'pages': pages,'genre':genre,'mood':mood, 'stars':stars,'reviews':reviews,'series':series,
        'mix':mix,'character':character,'plot':plot}

#code from terminal:
#scrapy crawl first -o scrap.json

#####scrapping the same amount of pages took less then a minute
