from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import os


def scrape_mediamarkt_links(pages):
    options = Options()

    driver = webdriver.Chrome(options=options)
    base_url = "https://www.mediamarkt.hu/hu/search.html?query=telefon&page="

    hrefs = set() # use a set to avoid duplicates

    for page in range(1, pages + 1):
        url = f"{base_url}{page}"
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/hu/product/_"]'))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        product_links = soup.select('a[href*="/hu/product/_"]')

        for link in product_links:
            href = link.get('href')
            if href:
                hrefs.add("https://www.mediamarkt.hu" + href)

    driver.quit()
    return list(hrefs)


def extract_product_data(url, driver):

    driver.get(url)
    time.sleep(3)  # Let JS load

    soup = BeautifulSoup(driver.page_source, "html.parser")
    data_dict = {}

    # Extract price
    price_whole = soup.find("span", attrs={"data-test": "branded-price-whole-value"})
    if price_whole:
        price_text = price_whole.get_text(strip=True)
        data_dict["price"] = price_text
    else:
        data_dict["price"] = None

    # Extract technical specifications from <tbody>
    tbodies = soup.find_all("tbody")
    for tbody in tbodies:
        rows = tbody.find_all("tr")
        for row in rows:
            tds = row.find_all("td")
            if len(tds) == 2:
                key = tds[0].get_text(strip=True)
                value = tds[1].get_text(strip=True)
                data_dict[key] = value

    # add product URL
    data_dict["URL"] = url

    # Extract reviews
    #data_dict["reviews"] = None  # Placeholder

    return data_dict



if __name__ == "__main__":
    product_links = scrape_mediamarkt_links(pages=2)
    print(f"Found {len(product_links)} product links.")

    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    all_products = []

    for i, link in enumerate(product_links):
        try:
            print(f"Scraping ({i+1}/{len(product_links)}): {link}")
            product_data = extract_product_data(link, driver)
            all_products.append(product_data)
        except Exception as e:
            print(f"Failed to scrape {link}: {e}")

    driver.quit()

    # Create DataFrame and export
    df = pd.DataFrame(all_products)
    
    os.makedirs("Data", exist_ok=True)
    output_path = os.path.join("Data", "mediamarkt_products.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Saved to {output_path}")
