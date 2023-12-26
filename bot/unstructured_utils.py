
from langchain.document_loaders import UnstructuredFileLoader
from bs4 import BeautifulSoup


class DocxLoader:
    def __init__(self, path):
        loader = UnstructuredFileLoader(path,  mode="elements")
        self.documents = loader.load()
    @staticmethod
    def get_text_from_html(html):

        # Parse the HTML content
        soup = BeautifulSoup(html, 'html.parser')

        # Find all occurrences of </td><td> and extract the text in between
        text_between_tags = set()  # Using a set to avoid duplicates
        for row in soup.find_all('tr'):
            tds = row.find_all('td')
            if len(tds) >= 2:
                for i in range(len(tds) - 1):  # Iterate up to the second-to-last element
                    text_between_tags.add(tds[i].get_text(strip=True))
        return  text_between_tags

    def __call__(self,):
        text_list = []
        for doc in self.documents:
            html = doc.metadata['text_as_html']
            text_list.extend(self.get_text_from_html(html))
        final_text = []
        for text in text_list:
            if len(text) > 75:
                final_text.append(text)
        return final_text

    